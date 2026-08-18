from manim import *
import numpy as np

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

class Section1Scene(TeachingScene):
    def construct(self):
        title_text = "The Mystery of AI Memory"
        lecture_lines = [
            "Transformers use attention for context and MLPs for memory.",
            "Attention is like a librarian finding relevant books.",
            "MLP layers store the actual facts and information."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_ATTENTION = BLUE_C
        COLOR_MLP = GREEN_C
        COLOR_HIGHLIGHT = YELLOW_C

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        # Attention block (using place_in_area) - Resolved Issue #30: B3 to B5
        attention_box = Rectangle(width=3.0, height=0.8, color=COLOR_ATTENTION, fill_opacity=0.3)
        attention_label = Text("Attention (Context)", font_size=20, color=COLOR_ATTENTION)
        attention_group = VGroup(attention_box, attention_label)
        self.place_in_area(attention_group, "B3", "B5")
        
        # MLP block (using place_in_area) - Resolved Issue #28: D3 to D5
        mlp_box = Rectangle(width=3.0, height=0.8, color=COLOR_MLP, fill_opacity=0.3)
        mlp_label = Text("MLP (Fact Memory)", font_size=20, color=COLOR_MLP)
        mlp_group = VGroup(mlp_box, mlp_label)
        self.place_in_area(mlp_group, "D3", "D5")
        
        self.play(FadeIn(attention_group), FadeIn(mlp_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Shift highlight to second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Librarian icon (SVG) - Resolved Issue #25
        try:
            librarian_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/librarian.svg")
            librarian_icon.set_color(WHITE)
        except:
            # Fallback if asset not found
            librarian_head = Circle(radius=0.15, color=WHITE, fill_opacity=0.5)
            librarian_body = Rectangle(width=0.4, height=0.4, color=WHITE, fill_opacity=0.5).shift(DOWN*0.3)
            librarian_icon = VGroup(librarian_head, librarian_body)
            
        librarian_label = Text("Librarian", font_size=14, color=WHITE)
        librarian_group = VGroup(librarian_icon, librarian_label).arrange(DOWN, buff=0.1)
        # Resolved Issue #30: Librarian at B1
        self.place_at_grid(librarian_group, "B1")
        
        # Arrow from Librarian to Attention
        arrow_lib = Arrow(librarian_group.get_right(), attention_group.get_left(), color=WHITE, stroke_width=2, buff=0.1)
        
        self.play(FadeIn(librarian_group), Create(arrow_lib))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shift highlight to third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Books icon (represented by a stack of rectangles)
        book_stack = VGroup(*[
            Rectangle(width=0.4, height=0.1, color=WHITE, fill_opacity=0.8).shift(UP*i*0.12)
            for i in range(3)
        ])
        books_label = Text("Facts", font_size=14, color=WHITE)
        books_group = VGroup(book_stack, books_label).arrange(DOWN, buff=0.1)
        # Resolved Issue #28: Books at D1
        self.place_at_grid(books_group, "D1")
        
        # Arrow from Books to MLP
        arrow_books = Arrow(books_group.get_right(), mlp_group.get_left(), color=WHITE, stroke_width=2, buff=0.1)
        
        self.play(FadeIn(books_group), Create(arrow_books))
        
        # Question: "What is the capital of France?"
        question = Text("What is the capital of France?", font_size=20)
        # Resolved Issue #29: Question at A2 to A5
        self.place_in_area(question, "A2", "A5")
        self.play(Write(question))
        
        # Highlight words "capital" and "France" in the question
        try:
            rect_capital = SurroundingRectangle(question[12:19], color=BLUE_A, buff=0.05)
            rect_france = SurroundingRectangle(question[23:29], color=BLUE_A, buff=0.05)
            self.play(Create(rect_capital), Create(rect_france))
        except:
            pass # Indexing robustness for Text mobjects
            
        # MLP Glow effect to indicate memory retrieval
        glow_rect = SurroundingRectangle(mlp_group, color=YELLOW_C, buff=0.1)
        self.play(Create(glow_rect))
        self.play(FadeOut(glow_rect), run_time=0.4)
        self.play(FadeIn(glow_rect), run_time=0.4)
        self.play(FadeOut(glow_rect), run_time=0.4)
        
        self.wait(3)
