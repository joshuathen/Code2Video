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
        # Setup data
        title_text = "The Hook: Same World, Different Languages"
        lecture_lines = [
            "- One point can be described by different landmarks.",
            "- A vector exists as a fixed point in space.",
            "- But its coordinates depend on the chosen grid."
        ]
        
        # Colors
        COLOR_MOUSE = "#FFD700"  # Yellow
        COLOR_LUNA = "#00FFFF"   # Cyan
        COLOR_ROBO = "#FF00FF"   # Magenta
        
        self.setup_layout(title_text, lecture_lines)

        # Assets
        mouse_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/mouse.svg"
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(COLOR_MOUSE))
        
        # Toy Mouse (the fixed point)
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/mouse.svg]
        # Resolve Issue 22 (Asset Integration) and Issue 25 (Scale 0.7)
        mouse = SVGMobject(mouse_path).set_color(COLOR_MOUSE)
        self.place_in_area(mouse, "C3", "D4", scale_factor=0.7)
        
        mouse_label = Text("Toy Mouse", font_size=16, color=COLOR_MOUSE)
        mouse_label.next_to(mouse, DOWN, buff=0.1)
        
        self.play(FadeIn(mouse), Write(mouse_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_LUNA)
        )
        
        # Luna's grid
        luna_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": COLOR_LUNA, "stroke_opacity": 0.3},
            axis_config={"stroke_color": COLOR_LUNA, "stroke_width": 2}
        )
        self.place_in_area(luna_grid, "A1", "F6", scale_factor=0.6)
        
        # Window Landmark
        window_icon = VGroup(
            Square(side_length=0.4, color=COLOR_LUNA),
            Line(start=[-0.2, 0, 0], end=[0.2, 0, 0], color=COLOR_LUNA),
            Line(start=[0, -0.2, 0], end=[0, 0.2, 0], color=COLOR_LUNA)
        )
        # Resolve Issue 23: Move to B5, scale 0.7
        self.place_at_grid(window_icon, "B5", scale_factor=0.7)
        window_label = Text("Window", font_size=14, color=COLOR_LUNA).next_to(window_icon, UP, buff=0.1)
        
        luna_stuff = VGroup(luna_grid, window_icon, window_label)
        self.play(FadeIn(luna_stuff))
        
        # Brief pulse of the mouse to show it is the object of interest
        self.play(mouse.animate.scale(1.2), run_time=0.3)
        self.play(mouse.animate.scale(1/1.2), run_time=0.3)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_ROBO)
        )
        
        # Robo-1's grid (rotated to show change of perspective)
        robo_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": COLOR_ROBO, "stroke_opacity": 0.3},
            axis_config={"stroke_color": COLOR_ROBO, "stroke_width": 2}
        ).rotate(35 * DEGREES)
        self.place_in_area(robo_grid, "A1", "F6", scale_factor=0.6)
        
        # Charger Landmark
        charger_icon = VGroup(
            Rectangle(height=0.4, width=0.3, color=COLOR_ROBO),
            Triangle(color=COLOR_ROBO).scale(0.1).rotate(180*DEGREES).move_to([0, 0.1, 0])
        )
        # Resolve Issue 24: Move to B2, scale 0.7
        self.place_at_grid(charger_icon, "B2", scale_factor=0.7)
        charger_label = Text("Charger", font_size=14, color=COLOR_ROBO).next_to(charger_icon, DOWN, buff=0.1)
        
        robo_stuff = VGroup(robo_grid, charger_icon, charger_label)
        
        # Transition from Luna's perspective to Robo's while keeping the mouse fixed
        self.play(
            ReplacementTransform(luna_stuff, robo_stuff),
            run_time=2
        )
        self.wait(2)
