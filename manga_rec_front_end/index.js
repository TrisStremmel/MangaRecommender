const express = require('express');
const mysql = require('mysql');
const app = express();
const session = require('express-session');
const {spawn} = require('child_process');
app.use(express.json());
const cors = require("cors")
app.use(cors())

app.use(session({
    secret: 'secret',
    resave: true,
    saveUninitialized: true,
}));

const con = mysql.createConnection({
  host: "washington.uww.edu",
  user: "hoganct11",
  password: "ch7463",
  database: "manga_rec"
});

con.connect(function(err) {
   if (err) throw err;
});

app.post('/auth/login', function(req, res){
    let username = req.body.username;
    let password = req.body.password;
    if (username && password){
        con.query('select userId from users where username=? and password = ?',[username, password], function(error, results){
           if (error) throw error;

           if(results.length > 0) {
               // req.session.loggedin = true;
               // req.session.username = username;
               // req.session.userId = results[0].userId;
               console.log('User',username,'with userId',results[0].userId);
               res.send(JSON.stringify(results[0].userId));
           } else {
               res.send('-1');
           }
        });
    }
});

app.post('/auth/register', function(req, res){
    let username = req.body.username;
    let password = req.body.password;
    if (username && password){
        con.query('call register(?,?)',[username, password], function(error, results){
            if (error) throw error;

            if(results.length > 0) {
                // req.session.loggedin = true;
                // req.session.username = username;
                // req.session.userId = results[0].userId;
                console.log('User',username,'with userId',results[0].userId);
                res.send(JSON.stringify(results[0].userId));
            } else {
                res.send('-1');
            }
        });
    }
});

app.get('/api/user/interest/:userId/:mangaId', function(req,res){
    let sql = 'select status from ratings where userId=? and mangaId=?';
    let sqlParams = [req.params.userId,req.params.mangaId];
    console.log(sql,sqlParams);
    con.query(sql,sqlParams,function (err,result) {
        if (err) throw err;
        else {
            if(!result.length)
                res.send(JSON.stringify('empty'));
            else if (result[0].status === 'interested') {
                res.send(true);
            } else if(result[0].status === 'not interested'){
                res.send(false);
            } else {
                res.send(JSON.stringify('empty'));
            }
        }
    });
});

app.post('/api/user/mal-cold-start/', function(req,res){
    let dataToSend;
    console.log("Getting user of", req.body.username, "with userID of", req.body.userId);
    // res.send(req.body.username);

    const python = spawn('C:/Users/Tris/Desktop/MangaRecommender/manga_rec_front_end/mangaRecommender/venv/Scripts/python', ['mangaRecommender/addUser.py', req.body.username, req.body.userId]);

    python.stdout.on('data', function (data) {
        dataToSend = data;
    });
    python.on('exit', (code) => {
        console.log("Done adding user from mal");
        console.log(dataToSend);
        res.send(dataToSend);
    });
});

app.get('/api/manga/:id', function(req, res) {
    let sql = 'select * from manga where id = ?'
    con.query(sql,req.params.id,function (err,result) {
        if (err) throw err;
        // console.log(result);
        res.send(JSON.stringify(result));
    });
    // res.send("id is set to " + req.params.id);
	// console.log("Request for manga id: " + req.params.id);
});

app.post('/api/search/', function(req, res) {
    let offset = 0;
    let limit = req.body.limit;
    const query = req.body.query;
    if(req.body.page !== undefined && req.body.page !== '0')
        offset = (req.body.page - 1) * limit;
    let sql = "call search(?,?,?)";
    let sqlParams = [query,limit,offset]
    // const sql = 'select title,id,pictureLink from manga where title like \'' + query + '\' limit ' + limit + ' offset ' + offset;
    // console.log(sql);
    console.log("Searching page",req.body.page,"with query", query);
    console.log(sql,sqlParams)
    con.query(sql,sqlParams,function (err,result) {
        if (err) throw err;
            // console.log(result);
            // console.log("Results:",JSON.stringify(result));
            res.send(JSON.stringify(result));
    })

});

app.get('/api/genre/:genre', function(req, res) {
    let sql = 'select * from manga where genre like ? order by rand() limit 10'
    let genre = '%'+req.params.genre+'%'
    con.query(sql, genre,function (err,result) {
        if (err) throw err;
        // console.log(result);
        res.send(result);
    });
    // res.send("id is set to " + req.params.id);
// console.log("Request for manga id: " + req.params.id);
});

app.get('/api/theme/:theme', function(req, res) {
    let sql = 'select * from manga where theme like ? order by rand() limit 10'
    let theme = '%'+req.params.theme+'%'
    con.query(sql, theme, function (err,result) {
        if (err) throw err;
        // console.log(result);
        res.send(result);
    });
    // res.send("id is set to " + req.params.id);
// console.log("Request for manga id: " + req.params.id);
});

app.post('/api/manga/interest', function(req,res) {
    // const sql = 'call Interest(' + InterestStatus + ',' + 'mangaId' + ',' + 'userId' + ')';
    console.log("User: " + req.body.userId + " is " + req.body.interest + " in manga with id " + req.body.mangaId);
    con.query('CALL Interest(?,?,?)',
        [
            req.body.interest,
            req.body.mangaId,
            req.body.userId
        ], function (err,result,field) {
        if (err) throw err;
        res.send(JSON.stringify(result));
    });
});

app.post('/api/manga/recommend', function(req,res){
    console.log('User of ' + req.body.userId + " is getting recommendations with filters " + req.body.filters);
    console.log('User of ' + req.body.userId + " is getting recommendations");
    let dataToSend;
	const command = "conda run -n for_manga_rec python mangaRecommender/recommender.py"

    const python = spawn('C:/Users/Tris/Desktop/MangaRecommender/manga_rec_front_end/mangaRecommender/venv/Scripts/python', ['mangaRecommender/recommender.py', req.body.userId, req.body.filters]);
    //const python = spawn('C:\Users\Tris Stremmel\anaconda3\envs\for_manga_rec\python', ['mangaRecommender/HelloWorld.py', "1"]);

    python.stdout.on('data', function (data) {
       dataToSend = data;
    });
    // python.stdout.on('data', data => {
    //    console.error(`stderr: ${data}`);
    // });
    python.on('exit', (code) => {
        console.log("Recommending done");
        console.log(`child process exited with code ${code}, ${dataToSend}`);
       res.send(dataToSend);
    });
});

app.get('/api/:user/manga/:id', function(req, res) {
    let sql = 'select * from ratings where userId = ? and mangaId = ?;'
    console.log("User of", req.session.userId);
    con.query(sql, [req.params.user, req.params.userId], function (err,result) {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
    // res.send("id is set to " + req.params.id);
// console.log("Request for manga id: " + req.params.id);
});

app.post('/api/:userId/manga/:mangaId/:interest', function(req,res) {
    // const sql = 'call Interest(' + InterestStatus + ',' + 'mangaId' + ',' + 'userId' + ')';
    console.log("User: " + req.params.userId + " is " + req.params.interest + " in manga with id " + req.params.mangaId);
    con.query('CALL Interest(?,?,?)',
        [
            req.params.interest,
            req.params.mangaId,
            req.params.userId
        ], function (err,result,field) {
            if (err) throw err;
            res.send(JSON.stringify(result));
        });
});

app.post('/api/:userId/:mangaId/:rating', function(req,res) {
    // const sql = 'call Interest(' + InterestStatus + ',' + 'mangaId' + ',' + 'userId' + ')';
    console.log("User: " + req.params.userId + " is " + req.params.rating+ " in manga with id " + req.params.mangaId);
    con.query('CALL AddRating(?,?,?)',
        [
            req.params.mangaId,
            req.params.userId,
            req.params.rating
        ], function (err,result,field) {
            if (err) throw err;
            res.send(JSON.stringify(result));
        });
});

app.get('/api/interest/:user', function(req, res) {
    let sql = 'select manga.* from manga inner join ratings on manga.id = ratings.mangaid where ratings.status = "interested" and ratings.userID = ? order by rand() limit 10'
    con.query(sql, req.params.user, function (err,result) {
        if (err) throw err;
        console.log(result);
        res.send(result);
    });
    // res.send("id is set to " + req.params.id);
// console.log("Request for manga id: " + req.params.id);
});

//PORT ENVIRONMENT VARIABLE
const port = process.env.PORT || 8080;
app.listen(port, () => console.log(`Listening on port ${port}..`));