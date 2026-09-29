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
        lines = [
            "Pigeonhole Principle bounds potential problem outcomes.",
            "Use extremal thinking to force a contradiction.",
            "Analyze the largest or smallest elements effectively.",
            "Find the most distant point from boundaries."
        ]
        self.setup_layout("Application: The Pigeonhole Principle & Extremal Methods", lines)
        
        # Paths for assets
        box_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg"
        pigeon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pigeon.svg"
        robot_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"

        # === Animation for Lecture Line 1 ===
        # Pigeonhole Principle bounds potential problem outcomes.
        self.lecture[0].set_color("#FF6347")
        holes = VGroup(*[SVGMobject(box_path, color="#87CEEB") for _ in range(3)]).arrange(RIGHT)
        pigeons = VGroup(*[SVGMobject(pigeon_path, color="#87CEEB") for _ in range(4)]).arrange(DOWN)
        
        self.place_at_grid(holes, 'B5', scale_factor=0.5)
        self.place_at_grid(pigeons, 'B6', scale_factor=0.5)
        self.play(FadeIn(holes), FadeIn(pigeons))

        # === Animation for Lecture Line 2 ===
        # Use extremal thinking to force a contradiction.
        self.lecture[1].set_color("#FF4500")
        pigeon_single = SVGMobject(pigeon_path, color="#FF6347")
        self.place_at_grid(pigeon_single, 'E3', scale_factor=0.6)
        self.play(FadeIn(pigeon_single))

        # === Animation for Lecture Line 3 ===
        # Analyze the largest or smallest elements effectively.
        self.lecture[2].set_color("#FF4500")
        label = Text("E_max", font_size=20, color="#FFD700")
        self.place_at_grid(label, 'E4', scale_factor=0.7)
        self.play(Write(label))

        # === Animation for Lecture Line 4 ===
        # Find the most distant point from boundaries.
        self.lecture[3].set_color("#FFFFFF")
        robot = SVGMobject(robot_path, color="#FFD700")
        self.place_at_grid(robot, 'D5', scale_factor=0.5)
        path = Line(start=self.grid['D5'], end=self.grid['E5'], color=WHITE)
        self.play(Create(robot), Create(path))
        self.play(robot.animate.move_to(self.grid['E5']))
        self.wait(1)
