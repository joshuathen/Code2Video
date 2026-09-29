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
        lecture_lines = ["Optimal guesses maximize expected information gain.", "The distribution shifts from flat to narrow.", "Higher certainty means fewer remaining possibilities.", "A robot calculates entropy for every word.", "Logic replaces intuition in this game."]
        self.setup_layout("Strategy: Maximizing Expected Information", lecture_lines)
        
        # Asset loading
        robot_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg"
        
        # === Animation for Lecture Line 1 ===
        # Show a tree structure representing guess branches. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg] (#FFFFFF)
        robot1 = SVGMobject(robot_asset).set_color(WHITE)
        self.place_at_grid(robot1, "B3", scale_factor=0.6)
        tree = VGroup(
            Dot(self.grid["C2"]), Dot(self.grid["C4"]), Dot(self.grid["C6"])
        ).set_color(WHITE)
        self.add(robot1, tree)
        self.lecture[0].set_color(WHITE)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Calculate expected value for information gain. (#FF0000)
        eq = MathTex(r"E[I] = \sum p \log \frac{1}{p}").set_color("#FF0000")
        self.place_in_area(eq, "B3", "C6", scale_factor=1.1)
        self.play(Write(eq))
        self.lecture[1].set_color("#FF0000")
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight the path with maximum information gain. (#00FF00)
        path = Line(self.grid["B3"], self.grid["C4"], color="#00FF00")
        self.add(path)
        self.lecture[2].set_color("#00FF00")
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Visualize the rapid reduction of remaining words. (#FFFF00)
        bars = VGroup(*[Rectangle(height=0.8, width=0.5, color="#FFFF00").move_to(self.grid[f"E{i}"]) for i in [2, 3, 4]])
        self.add(bars)
        self.lecture[3].set_color("#FFFF00")
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Compare heuristic vs entropy-based word selection using a processing bot. [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg] (#FFFFFF)
        robot2 = SVGMobject(robot_asset).set_color(WHITE)
        self.place_at_grid(robot2, "E5", scale_factor=0.6)
        text_comp = Text("Heuristic vs Entropy", font_size=20, color=WHITE)
        self.place_at_grid(text_comp, "E3", scale_factor=0.9)
        self.add(robot2, text_comp)
        self.lecture[4].set_color(WHITE)
        self.wait(2)
