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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Objective: Defining the 'Error'", [
            "The Cost Function measures prediction error.",
            "It acts as a scoreboard for AI.",
            "High error means poor performance."
        ])

        # === Animation for Lecture Line 1 ===
        # The Cost Function measures prediction error.
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        graph = axes.plot(lambda x: (x-1.5)**2 + 0.5, color="#00FF00")
        
        # Fix: Line 58, 59: self.place_in_area(axes, 'C1', 'F6', scale_factor=0.5)
        self.place_in_area(axes, "C1", "F6", scale_factor=0.5)
        self.place_in_area(graph, "C1", "F6", scale_factor=0.5)
        
        self.play(Create(axes), Create(graph))
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 2 ===
        # It acts as a scoreboard for AI.
        # Asset integration
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/scoreboard.svg"
        if os.path.exists(asset_path):
            scoreboard_icon = SVGMobject(asset_path)
        else:
            # Fallback if asset file not found
            scoreboard_icon = Text("Scoreboard", color=WHITE, font_size=24)
            
        # Fix: Line 67: self.place_at_grid(label, 'B4', scale_factor=0.7)
        self.place_at_grid(scoreboard_icon, "B4", scale_factor=0.7)
        self.play(FadeIn(scoreboard_icon))
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 3 ===
        # High error means poor performance.
        # Fix: Line 74: self.place_in_area(diff_line, 'C1', 'F6', scale_factor=0.5)
        diff_line = Line(start=axes.c2p(1.5, 0.5), end=axes.c2p(1.5, 2.5), color="#FF0000")
        self.place_in_area(diff_line, "C1", "F6", scale_factor=0.5)
        
        self.play(Create(diff_line))
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.wait(1)
