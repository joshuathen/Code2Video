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
        self.setup_layout("Prerequisite: The Concept of Invariance", [
            "An invariant remains unchanged under specific operations.",
            "Observe systems deforming while key properties stay fixed.",
            "The chameleon problem: Parity as an invariant."
        ])
        
        # Asset path
        chameleon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/chameleon.png"
        
        # === Animation for Lecture Line 1 ===
        # Display formula: I(s) = C. Label invariant: #FF00FF.
        formula = MathTex("I(s) = C", font_size=42)
        self.place_at_grid(formula, "B2")
        chameleon_icon = ImageMobject(chameleon_path)
        self.place_at_grid(chameleon_icon, "B3", scale_factor=0.15)
        
        self.play(Write(formula), FadeIn(chameleon_icon))
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        
        # === Animation for Lecture Line 2 ===
        # Visualize state transition: S1 -> S2. Arrow color: #00FFFF.
        s1 = Circle(radius=0.5, color=WHITE).set_fill(BLUE, opacity=0.5)
        s2 = Rectangle(height=0.8, width=0.8, color=WHITE).set_fill(GREEN, opacity=0.5)
        self.place_at_grid(s1, "C2", scale_factor=0.8)
        self.place_at_grid(s2, "C5", scale_factor=0.8)
        arrow = Arrow(s1.get_right(), s2.get_left(), color="#00FFFF")
        self.play(FadeIn(s1), Create(arrow), FadeIn(s2))
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # === Animation for Lecture Line 3 ===
        # Highlight invariant stability under transformation. Color: #FF4500.
        chameleon_icon2 = ImageMobject(chameleon_path)
        self.place_in_area(chameleon_icon2, "C4", "E6", scale_factor=0.25)
        self.play(FadeIn(chameleon_icon2))
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        
        self.wait(2)
