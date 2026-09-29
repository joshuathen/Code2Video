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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Query-Key-Value (QKV) Framework", [
            "Query is what you're searching for.",
            "Keys act like labels on books.",
            "Values contain the actual information retrieved."
        ])
        
        # Objects
        query = Rectangle(height=0.8, width=1.5, color="#FFD700", fill_opacity=0.6)
        key_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        key = Rectangle(height=0.8, width=1.5, color="#00CED1", fill_opacity=0.6)
        value = Rectangle(height=0.8, width=1.5, color="#FF69B4", fill_opacity=0.6)
        
        q_label = Text("Query", font_size=20)
        k_label = Text("Key", font_size=20)
        v_label = Text("Value", font_size=20)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.place_in_area(query, 'A1', 'A3', scale_factor=0.6)
        self.place_at_grid(q_label, 'A2', scale_factor=0.5)
        self.play(FadeIn(query), Write(q_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00CED1"))
        self.place_in_area(key, 'A4', 'A6', scale_factor=0.6)
        self.place_at_grid(k_label, 'A5', scale_factor=0.5)
        # Place key icon near key
        key_icon.scale(0.3).next_to(key, DOWN, buff=0.1)
        self.play(FadeIn(key), Write(k_label), FadeIn(key_icon))
        
        # Animate interaction
        self.play(query.animate.move_to(self.grid['B3']))
        dot_product = MathTex(r"Q \cdot K", font_size=24).next_to(query, DOWN)
        self.play(Write(dot_product))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF69B4"))
        self.place_in_area(value, 'C1', 'C3', scale_factor=0.6)
        self.place_at_grid(v_label, 'C2', scale_factor=0.5)
        self.play(FadeIn(value), Write(v_label))
        self.wait(2)
