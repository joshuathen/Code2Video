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
        # Initial Setup
        title_text = "The Hook: The Coastline Paradox"
        lecture_lines = [
            "How long is the coast of Great Britain?",
            "Smaller rulers reveal more jagged, hidden details.",
            "The measured length grows as we zoom in."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Straight Line (Road) - White
        road = Line(LEFT, RIGHT, color="#FFFFFF")
        # Issue 46: scale_factor=1.0
        self.place_in_area(road, "A1", "B3", scale_factor=1.0)
        road_label = Text("Straight Road", font_size=18, color="#FFFFFF")
        self.place_at_grid(road_label, "C2")
        
        # Jagged Coastline - Green
        coast_pts = [
            np.array([x, 0.2 * np.sin(x * 4) + 0.1 * np.cos(x * 12), 0]) 
            for x in np.linspace(-1.2, 1.2, 12)
        ]
        coast = VMobject(color="#00FF00").set_points_as_corners(coast_pts)
        # Issue 46: scale_factor=1.0
        self.place_in_area(coast, "A4", "B6", scale_factor=1.0)
        coast_label = Text("Coastline", font_size=18, color="#00FF00")
        self.place_at_grid(coast_label, "C5")
        
        self.play(Create(road), Write(road_label))
        self.play(Create(coast), Write(coast_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Magnifying area on the original coast
        zoom_rect = Square(side_length=0.6, color="#FFFF00")
        self.place_at_grid(zoom_rect, "B5")
        
        # Detailed Coastline (Zoomed in version) - Green
        # Increased frequency and amplitude of detail
        detailed_pts = [
            np.array([x, 0.4 * np.sin(x * 8) + 0.2 * np.cos(x * 16) + 0.1 * np.sin(x * 40), 0]) 
            for x in np.linspace(-1.5, 1.5, 80)
        ]
        detailed_coast = VMobject(color="#00FF00").set_points_as_corners(detailed_pts)
        # Issue 44: area "D2" to "E5", scale_factor=1.1
        self.place_in_area(detailed_coast, "D2", "E5", scale_factor=1.1)
        
        # Visual indicator of zoom
        zoom_arrow = Arrow(self.grid["C5"], self.grid["D4"], color="#FFFF00", buff=0.1)
        
        self.play(Create(zoom_rect))
        self.play(GrowArrow(zoom_arrow))
        self.play(TransformFromCopy(coast, detailed_coast))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        # Ant character - Issue 43: Asset integration
        ant = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ant.svg").scale(0.15)
        ant.set_color("#FFFF00")
        ant.move_to(detailed_coast.get_start())
        
        # Length counter display
        length_val = DecimalNumber(0, num_decimal_places=2, color="#FFFF00", font_size=24, mob_class=Text)
        length_text = Text("Measured Length:", font_size=18, color="#FFFF00")
        length_group = VGroup(length_text, length_val).arrange(RIGHT, buff=0.2)
        # Issue 45: scale_factor=0.8
        self.place_at_grid(length_group, "F4", scale_factor=0.8)

        self.play(FadeIn(ant), FadeIn(length_group))
        
        # Animate the ant moving and simulated length increasing
        self.play(
            MoveAlongPath(ant, detailed_coast),
            UpdateFromFunc(length_val, lambda m: m.set_value(np.linalg.norm(ant.get_center() - detailed_coast.get_start()) * 8.5)),
            run_time=6,
            rate_func=linear
        )
        self.wait(2)
