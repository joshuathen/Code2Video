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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Vector as a State", [
            "In physics, we represent a system's state as a vector.",
            "North and East are two distinct, base directions.",
            "A diagonal arrow combines both directions into one state."
        ])
        
        # Initial state: lecture lines dimmed
        for line in self.lecture:
            line.set_color(GRAY)

        # Asset Paths
        north_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/north.svg"
        compass_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg"

        # Position Anchor
        origin_pos = self.grid['D2']

        # === Animation for Lecture Line 1 ===
        # Line: "In physics, we represent a system's state as a vector."
        # Action: Draw a white arrow (#FFFFFF) pointing up [Asset: north.svg], labeled '|0>'.
        
        # Using SVGMobject for the North arrow asset
        north_icon = SVGMobject(north_path, color=WHITE)
        self.place_in_area(north_icon, 'B2', 'D2', scale_factor=0.8)
        # Shift icon so it looks like it originates near the origin and points to B2
        north_icon.shift(UP * 0.5) 
        
        label_0 = MathTex("|0\\rangle", color=WHITE)
        # Issue 32 & 43: Fix label_0 position to 'B2'
        self.place_at_grid(label_0, 'B2', scale_factor=0.8)
        label_0.shift(UP * 0.4) # Avoid direct overlap with the icon tip

        self.play(
            self.lecture[0].animate.set_color(WHITE),
            DrawBorderThenFill(north_icon),
            Write(label_0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "North and East are two distinct, base directions."
        # Action: Draw a second white arrow (#FFFFFF) pointing right, labeled '|1>'.
        
        east_tip = self.grid['D4']
        arrow_1 = Arrow(start=origin_pos, end=east_tip, color=WHITE, buff=0)
        
        label_1 = MathTex("|1\\rangle", color=WHITE)
        # Issue 33 & 43: Fix label_1 position to 'E5'
        self.place_at_grid(label_1, 'E5', scale_factor=0.8)
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            Create(arrow_1),
            Write(label_1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "A diagonal arrow combines both directions into one state."
        # Action: Fade in a yellow arrow (#FFFF00) pointing diagonally between the two basis arrows, 
        # using a compass [Asset: compass.svg] as a reference frame.
        
        compass = SVGMobject(compass_path)
        # Place compass centered at the origin of our vector system as a reference frame
        self.place_at_grid(compass, 'D2', scale_factor=2.5)
        compass.set_z_index(-1) # Ensure it stays behind arrows
        
        diagonal_tip = self.grid['B4']
        arrow_2 = Arrow(start=origin_pos, end=diagonal_tip, color=YELLOW, buff=0)
        
        self.play(
            self.lecture[2].animate.set_color(YELLOW),
            FadeIn(compass),
            FadeIn(arrow_2),
            run_time=2
        )
        self.wait(2)
