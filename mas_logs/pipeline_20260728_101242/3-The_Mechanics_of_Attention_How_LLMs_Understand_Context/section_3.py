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
        # Colors
        Q_COLOR = "#FF0000"
        K_COLOR = "#0000FF"
        V_COLOR = "#00FF00"
        BANK_COLOR = "#FFFFFF"

        lecture_lines = [
            "Each word transforms into three specialized vectors.",
            "Query asks: What information am I looking for?",
            "Key says: Here is what I offer others.",
            "Value holds the actual content being shared."
        ]

        self.setup_layout("The Triple Identity: Query, Key, and Value", lecture_lines)

        # Pre-creating Assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg]
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg").set_color(WHITE)
        
        # Barcode (Procedural ISBN represention)
        barcode = VGroup(*[Line(UP * 0.2, DOWN * 0.2, color=WHITE, stroke_width=2).shift(RIGHT * 0.06 * i) for i in range(7)])
        
        # Book (Procedural)
        book_rect = Rectangle(height=0.4, width=0.3, color=WHITE)
        book_lines = VGroup(*[Line(LEFT*0.1, RIGHT*0.1, color=WHITE, stroke_width=1).shift(UP*0.08*i) for i in range(-1, 2)])
        book = VGroup(book_rect, book_lines)

        # Labels
        q_label = Text("Query", font_size=20, color=Q_COLOR)
        k_label = Text("Key", font_size=20, color=K_COLOR)
        v_label = Text("Value", font_size=20, color=V_COLOR)

        # Boxes
        q_box = VGroup(Square(side_length=0.8, color=Q_COLOR), Text("Q", font_size=24, color=Q_COLOR))
        k_box = VGroup(Square(side_length=0.8, color=K_COLOR), Text("K", font_size=24, color=K_COLOR))
        v_box = VGroup(Square(side_length=0.8, color=V_COLOR), Text("V", font_size=24, color=V_COLOR))

        # === Animation for Lecture Line 1 ===
        # Each word transforms into three specialized vectors.
        # Requirement: Use columns 2, 4, 6 for spacing (Issue 37)
        self.lecture[0].set_color(WHITE)
        bank_rect = Rectangle(width=1.5, height=0.8, color=BANK_COLOR)
        bank_text = Text("Bank", font_size=24, color=BANK_COLOR)
        bank_group = VGroup(bank_rect, bank_text)
        self.place_at_grid(bank_group, "C4") # Center aligned
        
        self.play(FadeIn(bank_group))
        self.wait(1)

        # Split into Q, K, V at spread out positions
        self.place_at_grid(q_box, "C2")
        self.place_at_grid(k_box, "C4")
        self.place_at_grid(v_box, "C6")

        self.play(
            FadeOut(bank_group),
            FadeIn(q_box),
            FadeIn(k_box),
            FadeIn(v_box)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Query asks: What information am I looking for?
        self.lecture[1].set_color(Q_COLOR)
        self.place_at_grid(magnifying_glass, "B2", scale_factor=0.6)
        
        self.play(
            Indicate(q_box, color=WHITE),
            FadeIn(magnifying_glass)
        )
        self.play(Write(self.place_at_grid(q_label, "D2")))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Key says: Here is what I offer others.
        self.lecture[2].set_color(K_COLOR)
        self.place_at_grid(barcode, "B4", scale_factor=0.8)
        
        self.play(
            Indicate(k_box, color=WHITE),
            FadeIn(barcode)
        )
        self.play(Write(self.place_at_grid(k_label, "D4")))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Value holds the actual content being shared.
        self.lecture[3].set_color(V_COLOR)
        self.place_at_grid(book, "B6", scale_factor=0.8)
        
        self.play(
            Indicate(v_box, color=WHITE),
            FadeIn(book)
        )
        self.play(Write(self.place_at_grid(v_label, "D6")))
        self.wait(2)

        # Final pause to observe the identity spread
        self.wait(3)
