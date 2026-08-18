from manim import *
import os

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
        self.setup_layout("Geometric Intuition: The Parallelogram Expansion", [
            "Varying vectors change the parallelogram's shape.",
            "The cross product length scales with this area.",
            "Wider angles create longer perpendicular vectors."
        ])
        
        # Asset path
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg"
        
        # Create parallelogram asset
        if os.path.exists(asset_path):
            parallelogram = SVGMobject(asset_path)
        else:
            # Fallback
            parallelogram = Polygon(ORIGIN, RIGHT*1.5, RIGHT*1.5+UP*0.5, UP*0.5, 
                                    fill_color=BLUE, fill_opacity=0.3, stroke_color=WHITE)
        
        # Create vectors
        vec_a = Vector(LEFT*0.5 + UP*1.0, color=BLUE)
        vec_b = Vector(RIGHT*1.0 + UP*0.2, color=GREEN)
        group = VGroup(vec_a, vec_b, parallelogram)
        
        # Position group (Addressing Issue 27 and 29)
        self.place_in_area(group, 'A2', 'C4', scale_factor=0.9)
        
        # Cross product vector
        cp_vec = Vector(OUT*1.5, color=YELLOW)
        
        # Position cp_vec (Addressing Issue 28 and 29)
        self.place_at_grid(cp_vec, 'C5', scale_factor=0.75)

        # === Animation for Lecture Line 1 ===
        # Show parallelogram
        self.play(FadeIn(group), self.lecture[0].animate.set_color("#FF33A1"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform parallelogram and highlight area
        self.play(
            parallelogram.animate.set_color("#FF33A1"),
            self.lecture[1].animate.set_color("#33FF57")
        )
        self.play(GrowArrow(cp_vec))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Animate shearing effect
        self.play(
            vec_b.animate.rotate(0.3, about_point=ORIGIN), 
            parallelogram.animate.rotate(0.3, about_point=ORIGIN),
            cp_vec.animate.set_length(2.5),
            self.lecture[2].animate.set_color("#33A1FF")
        )
        self.wait(2)
