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
        # Title and Lecture lines from Storyboard
        title_text = "The Hook: The Cheetah's Instantaneous Speed"
        lecture_lines = [
            "Dash the cheetah sprints across the savanna.",
            "Average speed is calculated over total time.",
            "But how fast is he at this exact moment?"
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors for each stage
        color1 = "#FFD700"  # Gold for the Cheetah sprint
        color2 = "#87CEEB"  # Sky Blue for Average Speed context
        color3 = "#00FF00"  # Lime Green for Instantaneous Speed highlight

        # Asset path (Issue 25)
        cheetah_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg"

        # Visual elements setup
        # Issue 25: Load cheetah using SVGMobject
        cheetah = SVGMobject(cheetah_asset).set_color(color1).scale(0.3)
        
        # Using columns 2-5 for the track (C2 to C5) to maintain buffer
        path = Line(self.grid["C2"], self.grid["C5"], color=GREY_B)
        
        # Issue 30: Position path_label in E3-E4 area
        path_label = Text("100m Track", font_size=16, color=GREY_B)
        self.place_in_area(path_label, "E3", "E4", scale_factor=0.8)

        # ValueTracker for persistent movement (L008 renderer time equivalent for props)
        t_tracker = ValueTracker(0)
        cheetah.add_updater(lambda m: m.move_to(path.point_from_proportion(t_tracker.get_value())))

        # === Animation for Lecture Line 1 ===
        # Dash the cheetah sprints across the savanna.
        self.lecture[0].set_color(color1)
        self.add(path, path_label)
        self.add(cheetah)
        self.play(t_tracker.animate.set_value(1), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Average speed is calculated over total time.
        self.lecture[1].set_color(color2)
        
        # Average speed visualization: brace and text
        brace = BraceBetweenPoints(self.grid["C2"], self.grid["C5"], UP, color=color2, buff=0.1)
        avg_label = Text("Average Speed = Total Dist / Total Time", font_size=18, color=color2)
        
        # Issue 29: Use area B1 to B6 for wide formula
        self.place_in_area(avg_label, "B1", "B6", scale_factor=0.8)
        
        self.play(Create(brace), Write(avg_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # But how fast is he at this exact moment?
        self.lecture[2].set_color(color3)
        
        # Freeze effect: Move back to mid-point (representing 2.5s mark snapshot)
        freeze_point = (self.grid["C2"] + self.grid["C5"]) / 2
        highlight_circle = Circle(radius=0.4, color=color3).move_to(freeze_point)
        
        instant_label = Text("Instantaneous Speed", font_size=20, color=color3)
        # Issue 31: Use area B2 to B5 for balance
        self.place_in_area(instant_label, "B2", "B5", scale_factor=0.8)

        self.play(
            FadeOut(brace),
            FadeOut(avg_label),
            t_tracker.animate.set_value(0.5), # Snap to snapshot position
            run_time=1.5
        )
        
        self.play(
            Create(highlight_circle),
            Write(instant_label),
            Indicate(cheetah, color=color3) # L004 highlight
        )
        self.wait(3)
