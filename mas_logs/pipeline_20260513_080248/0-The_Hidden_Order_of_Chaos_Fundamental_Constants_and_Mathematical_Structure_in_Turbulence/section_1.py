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
        # Initialize the layout with updated lecture lines and title
        self.setup_layout(
            "Introduction: The Beauty of the Unpredictable", 
            [
                "A moving object creates a complex wake of air.", 
                "This wake contains swirling vortices across multiple physical scales.", 
                "Kinetic energy transfers from large structures to tiny eddies.", 
                "A rigid mathematical skeleton exists beneath the apparent chaos.", 
                "Turbulence follows a universal and predictable hidden order."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(YELLOW)
        
        # Load train asset and position (Issue 25, Issue 29)
        train = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/train.svg")
        train.set_color("#A9A9A9")
        self.place_at_grid(train, "B2", scale_factor=0.8)
        
        # Glide train across the grid
        self.play(FadeIn(train))
        self.play(train.animate.move_to(self.grid["B6"]), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Transition highlight
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create white #FFFFFF spirals for air wake (Issue 25)
        spirals = VGroup()
        for pos_key in ["B2", "B3", "B4", "B5"]:
            s = VGroup(
                Arc(radius=0.15, start_angle=0, angle=1.5*PI, color="#FFFFFF"),
                Arc(radius=0.1, start_angle=PI, angle=PI, color="#FFFFFF")
            ).arrange(RIGHT, buff=-0.05)
            self.place_at_grid(s, pos_key, scale_factor=1.0)
            spirals.add(s)
            
        self.play(LaggedStart(*[FadeIn(s, shift=RIGHT) for s in spirals], lag_ratio=0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition highlight
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight/scale multi-scale spirals: blue #0000FF (large) and cyan #00FFFF (tiny)
        blue_spirals = VGroup()
        for pos in ["C2", "D2"]:
            s = VGroup(
                Arc(radius=0.3, start_angle=0, angle=1.5*PI, color="#0000FF"),
                Arc(radius=0.2, start_angle=PI, angle=PI, color="#0000FF")
            )
            self.place_at_grid(s, pos, scale_factor=1.2)
            blue_spirals.add(s)
            
        cyan_spirals = VGroup()
        for pos in ["C4", "C5", "D4", "D5"]:
            s = VGroup(
                Arc(radius=0.1, start_angle=0, angle=1.5*PI, color="#00FFFF"),
                Arc(radius=0.05, start_angle=PI, angle=PI, color="#00FFFF")
            )
            self.place_at_grid(s, pos, scale_factor=0.6)
            cyan_spirals.add(s)
            
        # Morph wake into defined multi-scale spirals
        self.play(
            ReplacementTransform(spirals, VGroup(blue_spirals, cyan_spirals)),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Transition highlight
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Morph into a rigid glowing green #00FF00 grid pattern
        green_grid = VGroup()
        # Horizontal grid lines
        for r in ["B", "C", "D", "E"]:
            line = Line(self.grid[f"{r}1"], self.grid[f"{r}6"], color="#00FF00", stroke_width=2)
            green_grid.add(line)
        # Vertical grid lines
        for c in ["1", "2", "3", "4", "5", "6"]:
            line = Line(self.grid[f"B{c}"], self.grid[f"E{c}"], color="#00FF00", stroke_width=2)
            green_grid.add(line)
        
        self.play(
            ReplacementTransform(VGroup(blue_spirals, cyan_spirals), green_grid),
            FadeOut(train),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Transition highlight
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Fade-in gold #FFD700 text: 'Hidden Mathematical Order' (Issue 30)
        order_text = Text("Hidden Mathematical Order", color="#FFD700", font_size=26, weight=BOLD)
        self.place_in_area(order_text, 'D2', 'E5', scale_factor=0.8)
        
        self.play(FadeIn(order_text))
        self.wait(3)
