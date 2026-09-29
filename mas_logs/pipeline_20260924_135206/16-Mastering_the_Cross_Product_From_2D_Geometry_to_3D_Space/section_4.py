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
        self.setup_layout("Geometric Meaning and Physical Applications", [
            "Magnitude of cross product equals |A||B|sin(theta).",
            "Torque is the physical application of this concept.",
            "Torque equals the radius vector times force vector."
        ])
        
        # Assets
        wrench = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wrench.svg")
        bolt = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bolt.svg")
        
        # Elements
        parallelogram = Polygon(
            ORIGIN, RIGHT*2 + UP*0.5, RIGHT*2.5 + UP*1.5, RIGHT*0.5 + UP*1.0,
            color="#FF8000"
        )
        fill = Polygon(
            ORIGIN, RIGHT*2 + UP*0.5, RIGHT*2.5 + UP*1.5, RIGHT*0.5 + UP*1.0,
            fill_opacity=0.3, fill_color="#808080", stroke_width=0
        )
        group = VGroup(fill, parallelogram)
        
        # Applying requested position constraints
        self.place_in_area(group, 'D3', 'F5', scale_factor=0.5)
        self.place_at_grid(wrench, 'B3', scale_factor=0.5)
        self.place_at_grid(bolt, 'B4', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF8000"))
        self.play(Create(parallelogram), FadeIn(fill), FadeIn(wrench))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#808080"))
        self.play(FadeIn(bolt))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(Rotate(group, angle=PI/4, about_point=bolt.get_center()))
        
        self.wait(2)
