from manim import *

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
        # Initialize the scene with title and lecture lines
        self.setup_layout("Prerequisite: The Binary Counter", [
            "Binary counting uses only zeros and ones.",
            "In three bits, we count from zero to seven.",
            "Notice how each bit toggles at a specific frequency."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        
        # Display the text '000' in the center with a light blue color (#ADD8E6).
        binary_val_text = Text("000", font_size=60, color="#ADD8E6")
        # Issue 23 Fix: Avoid overlap by moving initial text to D2-D5
        self.place_in_area(binary_val_text, "D2", "D5")
        
        self.play(Write(binary_val_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line and transition color
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#555555")
        )
        
        # Create 3 circles representing the bits. Initially all are dark gray (#555555).
        circles = VGroup(*[
            Circle(radius=0.35, fill_opacity=1, color="#555555", stroke_width=2) 
            for _ in range(3)
        ]).arrange(RIGHT, buff=0.6)
        
        # Issue 24 Fix: Position circles at E2-E5 to avoid overlapping with counter text later
        self.place_in_area(circles, "E2", "E5")
        
        # Replace the digits with the three circles
        self.play(
            ReplacementTransform(binary_val_text, circles)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Display the corresponding binary string above the circles
        # Positioned in C2-D5 (above the circles in E2-E5)
        counter_text = Text("000", font_size=48, color="#ADD8E6")
        self.place_in_area(counter_text, "C2", "D5")
        self.play(FadeIn(counter_text))
        self.wait(0.5)
        
        # Animate the circles toggling as the counter increments from 001 to 111
        for i in range(1, 8):
            bin_str = bin(i)[2:].zfill(3)
            
            # Prepare new text and bit colors
            new_text_mob = Text(bin_str, font_size=48, color="#ADD8E6")
            new_text_mob.move_to(counter_text.get_center())
            
            anims = [Transform(counter_text, new_text_mob)]
            
            for bit_idx, char in enumerate(bin_str):
                # Toggle color: bright yellow (#FFFF00) for 1, dark gray (#555555) for 0
                target_color = "#FFFF00" if char == '1' else "#555555"
                anims.append(circles[bit_idx].animate.set_color(target_color))
            
            self.play(*anims, run_time=0.6)
            self.wait(0.4)
            
        self.wait(3)
