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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Ground concepts in logical sequences.", "Define valid ranges clearly.", "Identify and explain logical boundaries."]
        self.setup_layout("Criterion 3: The 'So-What' Connection", lecture_lines)
        
        # Assets
        concept_box = Rectangle(width=2, height=1, color=BLUE).set_fill(BLUE, opacity=0.3)
        self.place_at_grid(concept_box, "B4", scale_factor=0.7)
        concept_label = Text("Abstract Concept", font_size=20).next_to(concept_box, UP)
        
        # Asset: bridge.svg
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_in_area(bridge, "B3", "C6", scale_factor=0.6)
        bridge.set_color(GREEN)
        
        reality_obj = Circle(radius=0.5, color=YELLOW).set_fill(YELLOW, opacity=0.3)
        self.place_at_grid(reality_obj, "E5", scale_factor=0.5)
        reality_label = Text("Reality", font_size=20).next_to(reality_obj, UP)
        
        connection_line = Line(concept_box.get_bottom(), reality_obj.get_top(), color=GREEN)
        
        # Using a simple checkmark shape as standard Manim
        checkmark = VGroup(
            Line(start=LEFT*0.2+DOWN*0.1, end=ORIGIN, color=GREEN),
            Line(start=ORIGIN, end=RIGHT*0.3+UP*0.3, color=GREEN)
        )
        self.place_at_grid(checkmark, "C4")
        checkmark.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(concept_box), FadeIn(concept_label))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show abstract idea linking to a real object
        self.play(FadeIn(bridge), FadeIn(reality_obj), FadeIn(reality_label), Create(connection_line))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(checkmark))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
