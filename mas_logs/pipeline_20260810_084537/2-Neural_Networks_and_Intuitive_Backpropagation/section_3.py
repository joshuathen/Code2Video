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
        self.setup_layout("Defining the Loss Function (The Yardstick)", [
            "Loss is the distance between guess and truth.",
            "Think of an archer hitting a target.",
            "Distance from the center defines our error."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show Mean Squared Error formula
        mse_formula = MathTex(r"L = (y - \hat{y})^2", color=WHITE)
        self.place_in_area(mse_formula, 'A2', 'C4', scale_factor=1.2)
        self.play(Write(mse_formula))
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 2 ===
        # Load assets
        archer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/archer.svg")
        target = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/target.svg")
        
        label_y = Text("y", font_size=24, color=BLUE)
        label_yhat = Text("ŷ", font_size=24, color=YELLOW)
        
        # Group them
        archer_group = VGroup(archer, label_y).arrange(DOWN)
        target_group = VGroup(target, label_yhat).arrange(DOWN)
        
        # Place adjacent for comparison
        full_group = VGroup(archer_group, target_group).arrange(RIGHT, buff=0.8)
        self.place_in_area(full_group, 'D2', 'F5', scale_factor=0.7)
        
        self.play(FadeIn(full_group))
        self.play(self.lecture[1].animate.set_color("#FFD700"))

        # === Animation for Lecture Line 3 ===
        # Highlight the distance with a line
        dist_line = DashedLine(archer_group.get_center(), target_group.get_center(), color="#FF4500", stroke_width=4)
        
        self.play(Create(dist_line))
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        
        self.wait(2)
