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

class Section2Scene(TeachingScene):
    def construct(self):
        # Define titles and lines
        title_text = "Prerequisite: General Position & Rotation"
        lecture_lines = [
            "We assume no three points lie on one line.",
            "This ensures the line hits only one point at once.",
            "The line's angle determines its unique position and state."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Use white for point-related concepts
        self.lecture[0].set_color(WHITE)
        
        # Points initially collinear (B4, C5, D6) to show the forbidden state
        dot1 = Dot(color=WHITE)
        dot2 = Dot(color=WHITE)
        dot3 = Dot(color=WHITE)
        
        # Following Issue 30, 31 fixes for better layout and avoiding overlap
        self.place_at_grid(dot1, 'B4', scale_factor=0.8)
        self.place_at_grid(dot2, 'C5', scale_factor=0.8)
        self.place_at_grid(dot3, 'D6', scale_factor=0.8)
        
        points_vg = VGroup(dot1, dot2, dot3)
        self.play(Create(points_vg), run_time=1.5)
        self.wait(1)
        
        # Move dot3 to E6 to break collinearity (General Position)
        self.play(dot3.animate.move_to(self.grid['E6']), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Cyan color for the windmill line
        self.lecture[1].set_color("#00FFFF")
        
        # Load line asset
        line_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/line.svg")
        line_asset.set_color("#00FFFF")
        line_asset.scale(2.5) # Scale to be long enough
        
        angle_tracker = ValueTracker(0)
        pivot_pos = dot2.get_center()
        
        # Use add_updater for rotation around pivot
        def rotate_line_around_pivot(mob):
            ang = angle_tracker.get_value()
            mob.move_to(pivot_pos)
            mob.set_angle(ang)
            
        line_asset.add_updater(rotate_line_around_pivot)
        self.play(FadeIn(line_asset))
        
        # Calculate angles to hit dot1 and dot3 to demonstrate unique hits
        v1 = dot1.get_center() - pivot_pos
        ang1 = np.arctan2(v1[1], v1[0])
        
        v3 = dot3.get_center() - pivot_pos
        ang3 = np.arctan2(v3[1], v3[0])
        
        # Sweep to hit dot1
        self.play(angle_tracker.animate.set_value(ang1), run_time=1.5)
        self.wait(0.5)
        # Sweep to hit dot3
        self.play(angle_tracker.animate.set_value(ang3), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Yellow for angle arc and Green for Pivot highlight
        self.lecture[2].set_color("#FFFF00")
        
        # Load pivot asset
        pivot_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pivot.svg")
        pivot_asset.set_color("#00FF00")
        pivot_asset.scale(0.3)
        pivot_asset.move_to(pivot_pos)
        
        # Pivot label at B5 (Issue 32 fix)
        pivot_label = Text("Pivot", color="#00FF00", font_size=20)
        self.place_at_grid(pivot_label, 'B5', scale_factor=0.8)
        
        # Angle tracking arc
        # Arc is lightweight enough for always_redraw
        angle_arc_dynamic = always_redraw(lambda: Arc(
            radius=0.6,
            start_angle=0,
            angle=angle_tracker.get_value(),
            arc_center=pivot_pos,
            color="#FFFF00"
        ))
        
        self.play(
            FadeIn(pivot_asset),
            Write(pivot_label),
            FadeIn(angle_arc_dynamic)
        )
        
        # Final rotation to show the full range of states
        self.play(angle_tracker.animate.set_value(angle_tracker.get_value() + PI), run_time=3)
        self.wait(2)
