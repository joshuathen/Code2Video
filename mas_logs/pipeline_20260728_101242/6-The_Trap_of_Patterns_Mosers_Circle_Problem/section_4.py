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

class Section4Scene(TeachingScene):
    def construct(self):
        # Define lecture lines
        lecture_lines = [
            "Predict the number of regions for six points.",
            "The pattern suggests we should find thirty-two regions.",
            "Count the regions carefully as we add six points.",
            "Surprisingly, we only find thirty-one regions this time.",
            "Our perfect doubling pattern has finally been broken."
        ]
        
        # Setup layout
        self.setup_layout("The Great 'Gotcha': When $n=6$", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show a circle (Circle, #FFFFFF) with 6 points (Dot, #FF0000) on its boundary.
        self.lecture[0].set_color("#FFFFFF")
        circle = Circle(radius=1.8, color="#FFFFFF")
        self.place_in_area(circle, "B2", "E5")
        
        # Dots on the boundary - slightly offset to ensure max regions (avoid 3-line intersection at center)
        dots = VGroup()
        for i in range(6):
            # Non-uniform spacing to ensure no three chords intersect at a single point
            angles = [0.1, 1.0, 2.2, 3.1, 4.3, 5.5]
            point = circle.point_at_angle(angles[i])
            dot = Dot(point, color="#FF0000", radius=0.08)
            dots.add(dot)
            
        self.play(Create(circle))
        self.play(Create(dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw all chords connecting the 6 points rapidly (Line, #FFFF00).
        # Display the text '32?' in the center of the circle (Text, #00FFFF).
        self.lecture[1].set_color("#FFFF00")
        
        chords = VGroup()
        dot_points = [d.get_center() for d in dots]
        for i in range(6):
            for j in range(i + 1, 6):
                line = Line(dot_points[i], dot_points[j], color="#FFFF00", stroke_width=1.5)
                chords.add(line)
                
        prediction_text = Text("32?", color="#00FFFF")
        # Fix Issue 32: Place at A3-A4 to avoid overlap
        self.place_in_area(prediction_text, 'A3', 'A4', scale_factor=1.0)
        
        self.play(Create(chords, run_time=1.5, lag_ratio=0.1))
        self.play(Write(prediction_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Count and number each small region sequentially from 1 to 31 (Text, #00FF00).
        self.lecture[2].set_color("#00FF00")
        
        # Approximate interior points for counting the 31 regions
        # This is illustrative as the exact coordinates of the 31 regions depend on dot placement
        # We will distribute them within the circle's area (B2-E5)
        counting_grid_keys = [
            "B2", "B3", "B4", "B5", 
            "C2", "C3", "C4", "C5", 
            "D2", "D3", "D4", "D5", 
            "E2", "E3", "E4", "E5",
            "B1", "C1", "D1", "E1",
            "B6", "C6", "D6", "E6",
            "F2", "F3", "F4", "F5",
            "A2", "A5", "A6"
        ]
        
        region_counts = VGroup()
        for i in range(31):
            num_text = Text(str(i+1), font_size=16, color="#00FF00")
            # Map index to grid positions but keep them distinct
            grid_pos = counting_grid_keys[i % len(counting_grid_keys)]
            self.place_at_grid(num_text, grid_pos)
            # Add a small random jitter to avoid exact overlap if same grid key used
            num_text.shift(np.array([np.random.uniform(-0.2, 0.2), np.random.uniform(-0.2, 0.2), 0]))
            region_counts.add(num_text)
            
        self.play(AnimationGroup(*[FadeIn(t) for t in region_counts], lag_ratio=0.1, run_time=4))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Surprisingly, we only find thirty-one regions this time.
        self.lecture[3].set_color("#FF0000")
        
        truth_text = Text("31!", color="#FF0000", weight=BOLD)
        # Fix Issue 33: Place at A3-A4 to avoid overlap
        self.place_in_area(truth_text, 'A3', 'A4', scale_factor=1.0)
        
        # Issue 26: Integrate based.svg
        based_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg")
        based_icon.scale(0.3)
        # Position icon next to truth text
        based_icon.next_to(truth_text, RIGHT, buff=0.2)
        
        truth_group = VGroup(truth_text, based_icon)
        
        self.play(ReplacementTransform(prediction_text, truth_text))
        self.play(FadeIn(based_icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Our perfect doubling pattern has finally been broken.
        self.lecture[4].set_color("#FF0000")
        
        self.play(truth_group.animate.scale(1.2))
        self.wait(2)
