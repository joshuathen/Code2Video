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
        # 1. Setup layout with title and lecture lines
        title_text = "Translating the Necklace to Geometry"
        lecture_lines = [
            "Wrap the necklace around a transparent circular ring.",
            "A single diameter represents a cut across the loop.",
            "Rotating this diameter changes the bead counts continuously."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define specific colors for each animation stage to match lecture line highlights
        color1 = YELLOW   # For wrapping the necklace
        color2 = BLUE_B   # For the diameter cut
        color3 = GREEN_A  # For the dynamic rotation and counting

        # === Animation for Lecture Line 1 ===
        # Create a linear necklace with rubies (RED) and emeralds (GREEN)
        # Using 8 beads: 4 Red, 4 Green in an asymmetric pattern to show count variation
        bead_colors = [RED, RED, RED, GREEN, GREEN, GREEN, RED, GREEN]
        beads = VGroup(*[
            Circle(radius=0.15, fill_opacity=1.0, color=c, stroke_width=1, stroke_color=WHITE) 
            for c in bead_colors
        ])
        beads.arrange(RIGHT, buff=0.2)
        
        # Position the initial straight necklace in the middle-right area
        self.place_in_area(beads, "C1", "C6", scale_factor=0.9)
        
        # Color the lecture line
        self.play(self.lecture[0].animate.set_color(color1))
        self.play(Create(beads))
        self.wait(1)

        # Geometric target: Find the center of the right-side visualization area
        dummy_anchor = Square().scale(0.1)
        self.place_in_area(dummy_anchor, "B2", "E5")
        center_pt = dummy_anchor.get_center()
        
        # Create the transparent ring reference
        radius = 1.3
        ring = Circle(radius=radius, color=WHITE, stroke_opacity=0.3).move_to(center_pt)
        
        # Calculate positions on the circle for wrapping
        # We offset the start angle by 22.5 degrees (PI/8) so beads aren't split by horizontal/vertical diameters initially
        bead_angles = [PI/8 + i * (2*PI/8) for i in range(len(bead_colors))]
        circle_positions = [
            center_pt + radius * np.array([np.cos(ang), np.sin(ang), 0])
            for ang in bead_angles
        ]

        # Animate the transition from line to circle (wrapping the necklace)
        self.play(
            *[beads[i].animate.move_to(circle_positions[i]) for i in range(len(beads))],
            Create(ring),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(self.lecture[1].animate.set_color(color2))
        
        # Value tracker for the rotation angle of the diameter
        angle_tracker = ValueTracker(0)
        
        def get_diameter():
            ang = angle_tracker.get_value()
            # Draw a diameter that extends slightly past the ring for visibility
            p1 = center_pt + (radius + 0.5) * np.array([np.cos(ang), np.sin(ang), 0])
            p2 = center_pt + (radius + 0.5) * np.array([np.cos(ang + PI), np.sin(ang + PI), 0])
            return Line(p1, p2, color=WHITE, stroke_width=4)

        diameter_line = always_redraw(get_diameter)
        self.play(Create(diameter_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(self.lecture[2].animate.set_color(color3))

        # Helper function to count beads on either side of the diameter
        def get_counts():
            ang = angle_tracker.get_value() % (2 * PI)
            counts = {'AR': 0, 'AG': 0, 'BR': 0, 'BG': 0}
            for i, b_ang in enumerate(bead_angles):
                # Normalize angle difference to [0, 2PI)
                diff = (b_ang - ang) % (2 * PI)
                # One side of the diameter is the arc [0, PI) relative to the cut point
                if diff < PI:
                    if bead_colors[i] == RED: counts['AR'] += 1
                    else: counts['AG'] += 1
                else:
                    if bead_colors[i] == RED: counts['BR'] += 1
                    else: counts['BG'] += 1
            return counts

        # Dynamic labels that update as the diameter rotates, placed at grid top (A) and bottom (F)
        label_side_a = always_redraw(lambda: self.place_at_grid(
            Text(f"Side A: {get_counts()['AR']} Rubies, {get_counts()['AG']} Emeralds", 
                 font_size=18, color=color3),
            "A4"
        ))
        label_side_b = always_redraw(lambda: self.place_at_grid(
            Text(f"Side B: {get_counts()['BR']} Rubies, {get_counts()['BG']} Emeralds", 
                 font_size=18, color=color3),
            "F4"
        ))

        self.add(label_side_a, label_side_b)
        
        # Rotate the diameter full circle to show how bead counts change with the cut position
        self.play(angle_tracker.animate.set_value(2 * PI), run_time=8, rate_func=linear)
        self.wait(2)
