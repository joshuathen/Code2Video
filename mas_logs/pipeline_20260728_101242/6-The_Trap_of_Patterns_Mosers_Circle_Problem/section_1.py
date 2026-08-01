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
        lecture_lines = [
            "Divide a pizza by connecting points on the crust.",
            "Find the maximum slices created by these straight cuts.",
            "Two points and one cut create two separate regions."
        ]
        self.setup_layout("The Pizza Slicing Challenge", lecture_lines)
        
        # === Pizza setup with Asset ===
        # Use the provided pizza asset
        pizza_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg").set_color(WHITE)
        # Use a hidden circle for geometric reference (positioning dots on boundary)
        pizza_circle = Circle(radius=1.8).set_stroke(opacity=0)
        
        self.place_in_area(pizza_asset, "B2", "E5", scale_factor=2.0)
        pizza_circle.match_height(pizza_asset).move_to(pizza_asset)
        
        # Dots
        angle1 = 45 * DEGREES
        angle2 = 225 * DEGREES
        dot1 = Dot(pizza_circle.point_at_angle(angle1), color="#FF0000")
        dot2 = Dot(pizza_circle.point_at_angle(angle2), color="#FF0000")

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(DrawBorderThenFill(pizza_asset))
        self.play(FadeIn(dot1), FadeIn(dot2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FF00")
        )
        
        # One cut connecting the 2 points
        line12 = Line(dot1.get_center(), dot2.get_center(), color="#FFFF00")
        self.play(Create(line12))
        
        # Labels for 2 regions
        label_1 = Text("1", font_size=24, color="#00FF00")
        label_2 = Text("2", font_size=24, color="#00FF00")
        self.place_at_grid(label_1, "C5")
        self.place_at_grid(label_2, "D2")
        
        self.play(Write(label_1), Write(label_2))
        self.wait(2)
        
        # === Transition to 3-point case (as requested by storyboard) ===
        self.play(FadeOut(label_1), FadeOut(label_2))
        
        angle3 = 135 * DEGREES
        dot3 = Dot(pizza_circle.point_at_angle(angle3), color="#FF0000")
        
        # All possible lines for 3 points: (1,2), (1,3), (2,3)
        # line12 already exists
        line13 = Line(dot1.get_center(), dot3.get_center(), color="#FFFF00")
        line23 = Line(dot2.get_center(), dot3.get_center(), color="#FFFF00")
        
        self.play(FadeIn(dot3))
        self.play(Create(line13), Create(line23))
        
        # Label 4 regions - Fixing positions as per VideoCritic issues
        l1 = Text("1", font_size=24, color="#00FF00")
        l2 = Text("2", font_size=24, color="#00FF00")
        l3 = Text("3", font_size=24, color="#00FF00")
        l4 = Text("4", font_size=24, color="#00FF00")
        
        # Issue 28 fix: Move l2 from C5 to B5
        # Issue 29 fix: Move l3 from D2 to E2
        # Issue 30 fix: Move l4 from D4 to E4
        self.place_at_grid(l1, "B3", scale_factor=0.8)
        self.place_at_grid(l2, "B5", scale_factor=0.8)
        self.place_at_grid(l3, "E2", scale_factor=0.8)
        self.place_at_grid(l4, "E4", scale_factor=0.8)
        
        self.play(Write(VGroup(l1, l2, l3, l4)))
        self.wait(2)
        
        # Cleanup colors
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
