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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Third derivative is known as jerk.",
            "Jerk impacts motion control and engineering.",
            "Minimizing jerk ensures smooth robotic movement."
        ]
        self.setup_layout("Summary and Real-world Application", lecture_lines)
        
        # --- Create Visual Assets ---
        # 1. Jerk label
        jerk_label = Text("Jerk", font_size=36, color=RED)
        self.place_at_grid(jerk_label, 'A4', scale_factor=0.8)
        
        # 2. Robotic arm icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg]
        robot_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        if os.path.exists(robot_path):
            robot = SVGMobject(robot_path, color=GREEN)
        else:
            robot = Dot(color=GREEN) # Fallback
        self.place_at_grid(robot, 'D5', scale_factor=1.0)
        
        # 3. Graph
        axes = Axes(x_range=[0, 4, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.sin(x*np.pi), color=GREEN)
        graph = VGroup(axes, curve)
        self.place_in_area(graph, 'B4', 'E6', scale_factor=0.4)
        
        # --- Animation Sequence ---
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(jerk_label))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(Create(graph))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(FadeIn(robot))
        
        self.wait(2)
