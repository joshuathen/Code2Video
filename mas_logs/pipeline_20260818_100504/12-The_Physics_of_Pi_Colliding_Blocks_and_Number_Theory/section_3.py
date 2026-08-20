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
        self.setup_layout("The Geometric Mapping: Phase Space", ["Visualize motion in phase space.", "Map collisions to a wedge.", "Wedge angle depends on mass."])
        
        # === Animation for Lecture Line 1 ===
        # Draw a 2D coordinate grid in #CCCCCC.
        grid = Axes(x_range=[0, 6, 1], y_range=[0, 6, 1], x_length=5, y_length=5, axis_config={"color": "#CCCCCC"})
        self.place_in_area(grid, 'C3', 'F6', scale_factor=0.6)
        self.play(Create(grid), self.lecture[0].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 2 ===
        # Fade in a trajectory curve, create phase space point labeled 'P', move it.
        # Ensure we use grid coordinate mapping to keep consistent
        point_p = Dot(color="#FFFFFF")
        self.place_at_grid(point_p, 'A4', scale_factor=0.7)
        label_p = Text("P", font_size=20, color="#FFFFFF").next_to(point_p, UP, buff=0.1)
        p_group = VGroup(point_p, label_p)

        curve = ParametricFunction(lambda t: np.array([t, 0.5 * t**2, 0]), t_range=[0, 2], color="#00FF00")
        curve.scale(0.5).move_to(grid.get_center())
        
        # Grid labels for visual aid
        grid_labels = VGroup(*[Text(f"{i}", font_size=16) for i in range(1, 4)])
        self.place_in_area(grid_labels, 'C4', 'F6', scale_factor=0.5)
        
        self.play(FadeIn(curve), FadeIn(p_group), FadeIn(grid_labels), self.lecture[1].animate.set_color("#00FF00"))
        
        # Move P
        self.play(MoveAlongPath(p_group, curve), run_time=3)
        
        # Flash 'P' at the final coordinate
        self.play(Flash(point_p), self.lecture[2].animate.set_color("#00FF00"))
        self.wait(1)
