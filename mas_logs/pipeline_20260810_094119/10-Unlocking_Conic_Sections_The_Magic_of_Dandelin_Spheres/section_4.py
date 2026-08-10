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
        self.setup_layout("Parabolas and Hyperbolas", [
            "Angle changes create parabolas and hyperbolas.",
            "Parallel edges create the unique parabola shape.",
            "Cutting both nappes forms the beautiful hyperbola."
        ])
        
        # Assets
        plane = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        
        combined_group = VGroup(plane, cone)
        self.place_in_area(combined_group, 'B4', 'D6', scale_factor=0.6)

        # Representations of the cone-plane intersection
        parabola = VGroup(
            Line([-1, 1, 0], [1, -1, 0], color=WHITE),
            ArcBetweenPoints([-0.8, 1, 0], [0.8, 1, 0], angle=-PI, radius=0.8, color=WHITE)
        )
        hyperbola = VGroup(
            ParametricFunction(lambda t: np.array([t, 1/t, 0]), t_range=[0.5, 2], color=RED),
            ParametricFunction(lambda t: np.array([-t, -1/t, 0]), t_range=[0.5, 2], color=RED)
        )

        # Apply placements as per instructions
        self.place_at_grid(parabola, 'C4', scale_factor=0.8)
        self.place_at_grid(hyperbola, 'E5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(combined_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        # Rotate plane parallel to cone edge
        self.play(Rotate(plane, angle=PI/4), run_time=1)
        self.play(FadeIn(parabola))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        # Rotate plane to cut both nappes
        self.play(Rotate(plane, angle=PI/6), run_time=1)
        self.play(FadeIn(hyperbola))
        self.wait(2)
