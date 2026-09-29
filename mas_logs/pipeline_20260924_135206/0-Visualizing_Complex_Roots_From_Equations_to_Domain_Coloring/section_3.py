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
        lecture_lines = ["Iterative steps find complex roots.", "Derivatives guide us toward the target.", "The scout reaches the valley. [Asset: robot_scout]"]
        self.setup_layout("Numerical Root-Finding: Newton’s Method", lecture_lines)
        
        # Define objects
        graph = Axes(x_length=3, y_length=3, x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": False})
        curve = graph.plot(lambda x: 0.1 * x**3 - 0.5 * x, color=BLUE)
        point = Dot(graph.c2p(2, 0), color=BLUE)
        target = Dot(graph.c2p(0, 0), color=YELLOW)
        
        # Load asset
        robot_scout = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Place on grid (applying fixes from critics)
        self.place_in_area(VGroup(graph, curve), 'B3', 'E5', scale_factor=0.5)
        self.add(graph, curve, point)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(point))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        step = Line(graph.c2p(2, 0.3), graph.c2p(1, 0), color=RED)
        self.play(Create(step), point.animate.move_to(graph.c2p(1, 0)))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Applying fixes from critics
        self.place_at_grid(target, 'E4', scale_factor=0.6)
        self.place_at_grid(robot_scout, 'E5', scale_factor=0.5)
        
        self.play(FadeIn(target), FadeIn(robot_scout), point.animate.move_to(target.get_center()))
