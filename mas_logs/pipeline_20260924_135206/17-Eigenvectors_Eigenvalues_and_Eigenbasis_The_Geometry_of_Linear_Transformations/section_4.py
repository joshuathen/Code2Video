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
        self.setup_layout("Application: Principal Component Analysis (PCA)", [
            "PCA finds the eigenbasis of a dataset.",
            "It identifies directions of maximum variance.",
            "This simplifies complex movement tracking easily."
        ])
        
        # Load Assets
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")
        robot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Data cloud
        data_points = VGroup(*[Dot(point=np.array([np.random.normal(0, 0.5), np.random.normal(0, 0.2), 0])) for _ in range(50)])
        self.place_in_area(data_points, 'B3', 'E5', scale_factor=0.9)
        
        # Axes for PCA
        axes = Axes(x_length=3, y_length=3, axis_config={"include_tip": True})
        self.place_in_area(axes, 'B2', 'E4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(camera_icon, 'A2', scale_factor=0.5)
        self.play(FadeIn(data_points), FadeIn(axes), FadeIn(camera_icon))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        # Rotating axes to represent PC
        self.play(Rotate(axes, angle=PI/6))
        self.play(data_points.animate.rotate(PI/6))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.place_at_grid(robot_icon, 'F5', scale_factor=0.5)
        eigenvector_line = Line(start=np.array([-1.5, 0, 0]), end=np.array([1.5, 0, 0]), color="#FFFF00")
        self.place_at_grid(eigenvector_line, 'C3', scale_factor=1.0)
        self.play(Create(eigenvector_line), FadeIn(robot_icon))
        self.wait(2)
