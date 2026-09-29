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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Epicycle Intuition", [
            "Periodic motion combines simple circular rotations.",
            "Each circle moves at a unique frequency.",
            "Multiple circles trace out complex shapes."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Use [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg]
        pendulum = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg", color="#FF00FF")
        self.place_in_area(pendulum, "A3", "C5", scale_factor=0.5)
        self.play(Create(pendulum), self.lecture[0].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 2 ===
        # Using the fix from VideoCritic: place at D6 with scale 0.7
        c1 = Circle(radius=1.0, color="#00FFFF")
        self.place_at_grid(c1, "D6", scale_factor=0.7)
        vector1 = Line(c1.get_center(), c1.get_right(), color="#00FFFF")
        
        c2 = Circle(radius=0.5, color="#00FF00")
        c2.move_to(c1.get_right())
        vector2 = Line(c2.get_center(), c2.point_at_angle(PI/4), color="#00FF00")
        
        self.play(FadeIn(c1), Create(vector1), self.lecture[1].animate.set_color("#00FFFF"))
        self.play(FadeIn(c2), Create(vector2), self.lecture[2].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 3 ===
        # Use [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg]
        c3 = Circle(radius=0.25, color="#FFFF00")
        c3.move_to(c2.point_at_angle(PI/4))
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg", color="#FFFFFF")
        self.place_at_grid(gear, "E4", scale_factor=0.3)
        
        self.play(FadeIn(c3), FadeIn(gear), self.lecture[0].animate.set_color("#FFFFFF"), self.lecture[1].animate.set_color("#FFFFFF"), self.lecture[2].animate.set_color("#FFFFFF"))
