from manim import *
import numpy as np

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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Introduction: The Cheetah's Speed",
            [
                "Imagine a cheetah sprinting across the savanna.",
                "Total distance over time gives its average speed.",
                "The speedometer shows its instantaneous rate of change."
            ]
        )

        # Colors as per instructions
        GOLDEN = "#FFD700"
        SKY_BLUE = "#87CEEB"
        ORANGE_RED = "#FF4500"
        PURE_WHITE = "#FFFFFF"
        SOFT_GREY = "#AAAAAA"

        # === Animation for Lecture Line 1 ===
        # Imagine a cheetah sprinting across the savanna.
        self.play(self.lecture[0].animate.set_color(GOLDEN))
        
        cheetah_path = Line(self.grid["B2"], self.grid["B6"], color=SOFT_GREY)
        cheetah_path.set_stroke(opacity=0.3)
        self.add(cheetah_path)
        
        # Asset integration (Issue 19)
        # Using SVGMobject for the cheetah icon
        cheetah_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/che.svg").set_color(GOLDEN)
        cheetah_dot = Circle(radius=0.15, color=GOLDEN, fill_opacity=1.0)
        
        # Group dot and asset to move together
        cheetah_group = VGroup(cheetah_dot, cheetah_asset)
        self.place_at_grid(cheetah_group, "B2", scale_factor=0.6)
        
        self.play(Create(cheetah_group))
        self.play(cheetah_group.animate.move_to(self.grid["B6"]), run_time=3)
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Total distance over time gives its average speed.
        self.play(
            self.lecture[0].animate.set_color(PURE_WHITE),
            self.lecture[1].animate.set_color(SKY_BLUE)
        )
        
        # Add a path bracket below the path
        bracket = BraceBetweenPoints(self.grid["B2"], self.grid["B6"], direction=DOWN, color=SKY_BLUE)
        
        # Average Speed formula
        try:
            # Using MathTex for the formula
            formula = MathTex(
                r"\text{Average Speed} = \frac{\text{Total Distance}}{\text{Total Time}}",
                color=SKY_BLUE
            )
        except Exception:
            # Fallback to Text if MathTex fails (L022)
            formula = Text("Average Speed = Total Distance / Total Time", color=SKY_BLUE)

        # Apply fix from Issue 23: Position in area E2-F6 to avoid crowding
        self.place_in_area(formula, "E2", "F6", scale_factor=0.7)
        
        self.play(Create(bracket))
        self.play(Write(formula))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # The speedometer shows its instantaneous rate of change.
        self.play(
            self.lecture[1].animate.set_color(PURE_WHITE),
            self.lecture[2].animate.set_color(ORANGE_RED)
        )
        
        # Highlight a single point on the path (B4)
        inst_point = Dot(self.grid["B4"], color=ORANGE_RED, radius=0.1)
        
        # Label it "Instantaneous Speed"
        inst_label = Text("Instantaneous Speed", color=ORANGE_RED, font_size=22)
        # Apply fix from Issue 22: Position at C2 to avoid overlap with formula
        self.place_at_grid(inst_label, "C2", scale_factor=0.7)
        
        # Highlight the point
        self.play(Create(inst_point))
        self.play(Indicate(inst_point, color=ORANGE_RED))
        self.play(Write(inst_label))
        
        self.wait(2.0)
