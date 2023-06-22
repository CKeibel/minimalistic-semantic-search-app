import React, { useState, useEffect } from 'react';
import { Book } from './types';
import './App.css';

const App = () => {
  const [books, setBooks] = useState<Book[]>(); // Type
  const [query, setQuery] = useState<string>()

  const fetchData = async (route:string) => {
    const response = await (
      await fetch(route)
    ).json();
    setBooks(response)
  }

  const handleCancel = () => {
    setQuery("");
    fetchData("/all")
  }

  useEffect(() => {
    fetchData("/all")
  }, [])


  if (books == undefined)
  return (<p>Loading data...</p>);

  return (
    <div>
      <div className="search">
        <h1>Semantic Search</h1>
        <input className="searchbar" type="text" value={query} onChange={e => {setQuery(e.target.value)}} />
        <div className="actions">
          <button type="button" onClick={() => {fetchData(`/search?query=${query}`)}}>Search</button>
          <button type="button" onClick={handleCancel}>Cancel</button>
        </div>
      </div>
      <table className="results">
        {books.map(book => {
          return (
            <tr className="result-element">
              <td>{book.title}</td>
              <td>{book.author}</td>
              <td>{book.description}</td>
            </tr>
          );
        })}
      </table>
    </div>
  )


}

export default App;
