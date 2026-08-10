from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Attention uses three key components.",
            "Query: what I am looking for.",
            "Key: what I have to offer.",
            "Value: the actual information content.",
            "Library search analogy simplifies this process."
        ]
        self.setup_layout("The Mechanism: Queries, Keys, and Values", lecture_lines)
        
        # Setup labels
        q_label = Text("Query (Q)", color="#FF0000")
        k_label = Text("Key (K)", color="#00FF00")
        v_label = Text("Value (V)", color="#0000FF")
        
        # Setup Assets
        bookshelf = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bookshelf.svg")
        book = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Placing bookshelf for context
        self.place_at_grid(bookshelf, 'C3', scale_factor=0.5)
        self.play(FadeIn(bookshelf))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.place_at_grid(q_label, 'C2', scale_factor=0.7)
        self.play(FadeIn(q_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        self.place_at_grid(k_label, 'C3', scale_factor=0.7)
        self.play(FadeIn(k_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#0000FF")
        self.place_at_grid(v_label, 'C4', scale_factor=0.7)
        self.play(FadeIn(v_label))
        self.wait(1)
        
        # Animation of Q moving to K
        self.play(q_label.animate.move_to(k_label.get_center() + DOWN * 0.8))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        self.place_at_grid(book, 'E4', scale_factor=0.7)
        self.play(FadeIn(book))
        self.wait(2)
