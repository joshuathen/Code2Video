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

class Section5Scene(TeachingScene):
    def construct(self):
        # Mandatory layout setup with provided title and lecture lines
        lecture_lines = [
            "Limits range from visual paths to rigorous error bounds.",
            "Derivatives provide a shortcut for tough indeterminate forms.",
            "Mastery of limits builds the foundation for all calculus."
        ]
        self.setup_layout("Synthesis and Application", lecture_lines)

        # Colors for consistency with animation elements
        ANT_COLOR = "#00FF00"
        HOLE_COLOR = "#FF0000"
        EPSILON_COLOR = "#FFD700"
        DELTA_COLOR = "#00BFFF"
        FINAL_TEXT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Line: "Limits range from visual paths to rigorous error bounds."
        # Visual: Show the Robo-Ant crossing the graph's hole.
        self.play(self.lecture[0].animate.set_color(ANT_COLOR))
        
        # Position axes in the right-side grid system area
        # Resolved Issue 34: Reposition axes to avoid clutter
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-0.5, 4, 1],
            axis_config={"include_tip": False, "color": WHITE},
            x_length=4.5,
            y_length=3.5
        )
        self.place_in_area(axes, "B2", "E6", scale_factor=0.8)
        
        # Plot a curve with a hole at x=0
        graph = axes.plot(lambda x: x**2 + 1, x_range=[-2, 2], color=WHITE)
        
        # The Hole (pothole) visual representation
        hole_pos = axes.c2p(0, 1)
        hole_circle = Circle(radius=0.1, color=HOLE_COLOR).move_to(hole_pos)
        hole_fill = Dot(hole_pos, color=BLACK, radius=0.11)
        
        # Robo-Ant group (Ant + Label)
        # Resolved Issue 22: Integrate Robo-Ant asset
        ant_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ant.svg").set_color(ANT_COLOR).scale(0.3)
        ant_name = Text("Robo-Ant", font_size=16, color=ANT_COLOR).next_to(ant_asset, UP, buff=0.1)
        robo_ant = VGroup(ant_asset, ant_name)
        
        # Set starting position for the ant
        robo_ant.move_to(axes.c2p(-1.8, 1.8**2 + 1))
        
        self.play(Create(axes), Create(graph))
        self.play(Create(hole_circle), Create(hole_fill))
        self.add(robo_ant)
        
        # Moving along the path to simulate crossing the gap
        path = axes.plot(lambda x: x**2 + 1, x_range=[-1.8, 1.8])
        self.play(MoveAlongPath(robo_ant, path), run_time=4)

        # === Animation for Lecture Line 2 ===
        # Line: "Derivatives provide a shortcut for tough indeterminate forms."
        # Visual: Pulse the Epsilon-Delta box in the background representing the rigorous tool.
        self.play(self.lecture[1].animate.set_color(EPSILON_COLOR))
        
        # Epsilon-Delta "Safety Window" box around the hole
        ed_box = Rectangle(
            width=1.4, height=1.0, 
            color=EPSILON_COLOR, 
            fill_color=DELTA_COLOR, 
            fill_opacity=0.3
        ).move_to(hole_pos)
        
        # Derivative shortcut label (Visualizing the 'Derivative Speedometer' concept)
        deriv_shortcut = Text("dy/dx Shortcut", font_size=18, color=ANT_COLOR).next_to(robo_ant, DOWN, buff=0.1)
        
        self.play(FadeIn(ed_box), FadeIn(deriv_shortcut))
        self.play(
            ed_box.animate.scale(1.25),
            run_time=0.75,
            rate_func=there_and_back
        )
        self.play(
            ed_box.animate.scale(1.25),
            run_time=0.75,
            rate_func=there_and_back
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "Mastery of limits builds the foundation for all calculus."
        # Visual: Fade into final text 'Limits: The Destination'
        self.play(self.lecture[2].animate.set_color(FINAL_TEXT_COLOR))
        
        final_text_obj = Text("Limits: The Destination", font_size=36, color=FINAL_TEXT_COLOR)
        # Resolved Issue 35: Reposition and scale final text
        self.place_in_area(final_text_obj, "C2", "D5", scale_factor=0.9)
        
        self.play(
            FadeOut(axes), FadeOut(graph), FadeOut(hole_circle), 
            FadeOut(hole_fill), FadeOut(robo_ant), FadeOut(ed_box),
            FadeOut(deriv_shortcut)
        )
        self.play(Write(final_text_obj))
        self.wait(3)
