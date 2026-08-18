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
        lecture_lines = ["The robot agent plays Wordle.", "SLATE creates balanced outcome paths.", "Pathways show reduced possible words.", "Visualizing information gain in real-time.", "Balanced splits resolve puzzles faster."]
        self.setup_layout("Interactive Demonstration", lecture_lines)
        
        # UI Elements
        # Using the requested asset
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        word_ui = Rectangle(width=3, height=1, color=WHITE).set_fill(BLACK, opacity=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(robot, "C2", scale_factor=0.7)
        self.play(FadeIn(robot))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        slate_text = Text("SLATE", font_size=36, color=YELLOW)
        self.place_at_grid(slate_text, "B4")
        self.play(Write(slate_text))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        branch_lines = VGroup(*[Line(self.grid["B4"], self.grid[pos], color=GRAY) for pos in ["D2", "D4", "D6"]])
        self.play(Create(branch_lines))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF69B4"))
        info_gain = Text("Information Gained", font_size=20, color=PINK)
        self.place_at_grid(info_gain, "C5", scale_factor=0.75)
        # Using asset alongside info_gain
        self.play(FadeIn(info_gain))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        target = Text("Target Found", font_size=24, color=GREEN)
        self.place_at_grid(target, "E4", scale_factor=0.75)
        self.play(DrawBorderThenFill(target))
        
        self.wait(2)
