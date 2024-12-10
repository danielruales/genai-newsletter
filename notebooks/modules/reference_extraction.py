def create_content_from_df(df):
    """Convert dataframe rows into formatted content string."""
    all_content = '<articles>\n'
    all_content_list=[]
    all_content_dict = {}
    
    for idx, row in df.iterrows():
        # Format each article with consistent structure
        content = f"""
<START Article Number: {idx + 1}>
Title: {row['title']}
URL: {row['url']}
Summary: {row['summary']}
Description: {row['description']}
Created: {row['created_at'].strftime('%Y-%m-%d')}
Type: {row['media_type']}
<END Article Number: {idx + 1}>
"""
        # print('HERE***********', all_content_list)
        all_content += content
        all_content_list.append(content)
        all_content_dict[idx+1] = {
            "title": row['title'],
            "url": row['url'],
            "summary": row['summary'],
            "description": row['description'],
            "created_at": row['created_at'].strftime('%Y-%m-%d'),
            "media_type": row['media_type'],
            "content": content
        }
    all_content += '\n</articles>\n--------------------\n'
    
    return all_content, all_content_list, all_content_dict