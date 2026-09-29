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
        self.setup_layout("Boundary and Initial Conditions", [
            "PDEs require constraints for unique solutions.",
            "Initial conditions define the starting state.",
            "Boundary conditions constrain the edges."
        ])
        
        # Domain object
        domain = Rectangle(width=3, height=2, color=BLUE, fill_opacity=0.3)
        self.place_in_area(domain, "C3", "F6", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        # Highlight domain boundary edges
        boundary = SurroundingRectangle(domain, color="#FF4500", buff=0)
        self.play(self.lecture[0].animate.set_color("#FFD700"), Create(boundary))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show initial heat state distribution
        thermometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/thermometer.svg")
        self.place_at_grid(thermometer, "B3", scale_factor=0.5)
        
        heat_map = VGroup(*[Dot(point=domain.point_from_proportion(i/20), color=interpolate_color(BLUE, RED, i/20), radius=0.05) for i in range(21)])
        self.play(self.lecture[1].animate.set_color("#FF4500"), FadeIn(thermometer), FadeIn(heat_map))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualize boundary condition effect
        boundary_label = Text("T = 0°C", font_size=20, color=WHITE)
        self.place_at_grid(boundary_label, "B3", scale_factor=0.9)
        indicator = Arrow(start=boundary_label.get_bottom(), end=boundary.get_top(), color=WHITE, buff=0.1)
        
        self.play(self.lecture[2].animate.set_color("#00CED1"), Write(boundary_label), GrowArrow(indicator))
        self.wait(2)
