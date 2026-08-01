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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        title = "Visualizing Conditional Probability"
        lines = [
            "- Imagine the total sample space is a rectangle.",
            "- Knowing event B occurred shrinks our entire universe.",
            "- The new universe is just the area of B.",
            "- We look only at A's intersection with B.",
            "- This restricted area defines the new conditional probability."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Draw a large white rectangle (#FFFFFF) representing the total Sample Space.
        self.lecture[0].set_color(YELLOW)
        
        # Fixed: sample_space from B1 to F6 per Issue 36
        sample_space = Rectangle(width=5.0, height=4.0, color=WHITE, stroke_width=2)
        self.place_in_area(sample_space, "B1", "F6")
        
        # Fixed: ss_label at A1 per Issue 35
        ss_label = Text("S", font_size=24, color=WHITE)
        self.place_at_grid(ss_label, "A1", scale_factor=0.8)
        
        self.play(Create(sample_space), Write(ss_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Introduce two overlapping circles A (#FF00FF) and B (#00FFFF) inside the rectangle.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        circle_a = Circle(radius=0.8, color="#FF00FF", fill_opacity=0.3, stroke_width=3)
        circle_b = Circle(radius=0.8, color="#00FFFF", fill_opacity=0.3, stroke_width=3)
        
        # Fixed positions per Issue 36: circle_a at D3, circle_b at D4
        self.place_at_grid(circle_a, "D3")
        self.place_at_grid(circle_b, "D4")
        
        # Fixed positions per Issue 36: label_a at C3, label_b at C4
        label_a = Text("A", font_size=20, color="#FF00FF")
        label_b = Text("B", font_size=20, color="#00FFFF")
        self.place_at_grid(label_a, "C3")
        self.place_at_grid(label_b, "C4")

        self.play(
            FadeIn(circle_a), FadeIn(circle_b),
            Write(label_a), Write(label_b)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Darken the entire rectangle except for circle B, showing the "Shrinking Universe".
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Dark overlay with a hole for B. Fixed position per Issue 37: B1 to F6
        dark_overlay = Rectangle(width=5.0, height=4.0, fill_color=BLACK, fill_opacity=0.8, stroke_width=0)
        self.place_in_area(dark_overlay, "B1", "F6")
        
        # Cutout handles the hole
        universe_mask = Cutout(dark_overlay, circle_b, fill_opacity=0.8, color=BLACK, stroke_width=0)
        
        self.play(FadeIn(universe_mask))
        self.play(
            sample_space.animate.set_stroke(opacity=0.2),
            ss_label.animate.set_fill(opacity=0.2),
            circle_a.animate.set_stroke(opacity=0.2).set_fill(opacity=0.1),
            label_a.animate.set_fill(opacity=0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight the intersection area (A \cap B) with a bright yellow pulse (#FFFF00).
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        intersection = Intersection(circle_a, circle_b, color="#FFFF00", fill_opacity=0.9, stroke_width=4)
        self.play(FadeIn(intersection))
        self.play(Indicate(intersection, color="#FFFF00", scale_factor=1.1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Place a panda icon (#FFFFFF) within the intersection to represent the conditional event.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Issue 26: Use SVGMobject for asset integration
        # Issue 37: place_in_area D3 to D4, scale_factor=0.5
        panda_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/panda.svg", color=WHITE)
        self.place_in_area(panda_icon, "D3", "D4", scale_factor=0.5)
        
        self.play(DrawBorderThenFill(panda_icon))
        self.play(panda_icon.animate.scale(1.2), rate_func=there_and_back)
        self.wait(3)

        # Final cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
