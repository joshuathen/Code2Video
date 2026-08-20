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
        lecture_lines = [
            "A space needs two operations: Addition and Scalar Multiplication.",
            "Addition combines vectors, like adding geometric arrows together.",
            "Scalar multiplication scales vectors, changing their magnitude."
        ]
        self.setup_layout("The Foundation: The Two Core Operations", lecture_lines)
        
        # Load Assets
        # Using placeholder paths based on provided [Asset: ...]
        # Note: In real execution, these would be valid image files.
        # Since I am using SVGMobject/Arrow as placeholder for the asset logic, 
        # I'll create custom mobjects that mirror the visual requirement.
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Visual Addition using assets
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg
        vec1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg", color="#00FF00")
        vec2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg", color="#00FF00")
        sum_vec = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg", color="#FF0000")
        
        self.place_at_grid(vec1, "B4", scale_factor=0.5)
        # Position vec2 relative to vec1
        vec2.scale(0.5).next_to(vec1, RIGHT + UP, buff=0)
        
        self.play(FadeIn(vec1), FadeIn(vec2))
        
        # Highlight resultant
        self.place_at_grid(sum_vec, "C5", scale_factor=0.6)
        self.play(Create(sum_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(vec1), FadeOut(vec2), FadeOut(sum_vec))
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        scalar_vec = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg", color="#FFFF00")
        self.place_at_grid(scalar_vec, "E4", scale_factor=0.5)
        self.play(FadeIn(scalar_vec))
        self.play(scalar_vec.animate.scale(1.5).set_color("#FF0000"))
        self.wait(2)
