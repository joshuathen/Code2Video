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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Real-World Application: Machine Learning", [
            "High-dimensional spheres are practical tools in AI.",
            "Support Vector Machines use them for classification.",
            "They effectively separate complex data clusters."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show points in N-D space representing data from [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg]. Color #FFFFFF.
        server = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/server.svg", color=WHITE)
        feature_space = VGroup(*[
            Dot(color=WHITE, radius=0.05).move_to(np.random.uniform(-0.5, 0.5, 3))
            for _ in range(20)
        ], server)
        self.place_in_area(feature_space, 'A2', 'C5', scale_factor=0.5)
        self.play(FadeIn(feature_space))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Draw hyper-sphere separating points processed by [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]. Color #FFFFFF.
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color=WHITE)
        data_points = VGroup(*[
            Dot(color=WHITE, radius=0.08).move_to(np.random.uniform(-0.5, 0.5, 3))
            for _ in range(15)
        ], robot)
        self.place_in_area(data_points, 'D2', 'F6', scale_factor=0.6)
        self.play(FadeIn(data_points))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # Highlight volume density at margins. Color #FFD700.
        hyperplane = Square(side_length=2, color="#FFD700").set_fill(opacity=0.3)
        self.place_in_area(hyperplane, 'D3', 'F5', scale_factor=0.5)
        self.play(Create(hyperplane))
        self.lecture[2].set_color("#FFD700")
        
        self.wait(2)
