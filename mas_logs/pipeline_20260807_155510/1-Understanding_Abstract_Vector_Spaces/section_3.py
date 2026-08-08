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

class Section3Scene(TeachingScene, ThreeDScene):
    def construct(self):
        lecture_lines = ["Visualize states as abstract vectors.", "A robot's position is a vector.", "Configuration changes follow vector addition."]
        self.setup_layout("Visualizing Abstract Spaces", lecture_lines)
        
        # Create abstract structure (3D Axes representation)
        # Fixed axes obstruction (Issue 26, 37)
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"color": WHITE})
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.5)
        
        # Intersection Point and Label (Issue 28, 37)
        intersection_point = Dot(color=WHITE)
        intersection_point.move_to(axes.c2p(0, 0, 0))
        position_label = Text("Robot", font_size=20, color=WHITE)
        self.place_at_grid(position_label, 'C4', scale_factor=0.6)
        
        # Robot Asset
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'E3', scale_factor=0.4)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(axes))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.set_camera_orientation(phi=60 * DEGREES, theta=45 * DEGREES, run_time=2)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.play(
            intersection_point.animate.set_color("#FF00FF").scale(2),
            FadeIn(robot),
            FadeIn(position_label),
            run_time=1
        )
        self.add(intersection_point)
        self.wait(2)
