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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Classical Boundary: Prerequisite Knowledge", [
            "In our daily lives, objects exist in definite states.",
            "A light switch is either fully ON or OFF.",
            "This classical behavior is predictable and mutually exclusive."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a grey box (#888888) labeled 'Classical World' in the center.
        box = Rectangle(width=5.0, height=5.5, color="#888888", stroke_width=2)
        self.place_in_area(box, 'A1', 'F6')
        
        box_label = Text("Classical World", font_size=24, color="#888888")
        # Fix: Issue 20 - Use area-based placement for better centering
        self.place_in_area(box_label, 'A2', 'A4', scale_factor=1.0)
        box_label.shift(UP * 0.3)
        
        self.play(self.lecture[0].animate.set_color("#888888"))
        self.play(Create(box), Write(box_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the switch [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg] (#FFFFFF) inside the box in the 'OFF' position.
        # Issue 16: Integrate switch asset
        switch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg").set_color(WHITE)
        self.place_at_grid(switch, 'C3', scale_factor=0.8)
        
        # Rotate to indicate 'OFF' state initially
        switch.rotate(PI) 
        
        off_label = Text("OFF", font_size=24, color=WHITE)
        on_label = Text("ON", font_size=24, color=WHITE)
        self.place_at_grid(off_label, 'D4', scale_factor=1.0)
        self.place_at_grid(on_label, 'B4', scale_factor=1.0)
        
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(FadeIn(switch), FadeIn(off_label), FadeIn(on_label))
        self.wait(1)
        
        # Flip the switch to 'ON', adding a yellow glow (#FFFF00) to its label.
        glow = Circle(radius=0.6, color="#FFFF00", fill_opacity=0.4, stroke_width=0)
        # Fix: Issue 19 - Move glow to B3 to avoid overlap with ON label at B4
        self.place_at_grid(glow, 'B3', scale_factor=1.2)
        
        self.play(
            switch.animate.rotate(PI), # Flip to ON
            on_label.animate.set_color("#FFFF00"),
            FadeIn(glow),
            run_time=1
        )
        self.wait(1)
        
        # Flip back to OFF
        self.play(
            switch.animate.rotate(PI), # Flip back to OFF
            on_label.animate.set_color(WHITE),
            FadeOut(glow),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw a large red 'X' (#FF0000) over a hypothetical 'Both' label.
        both_label = Text("Both", font_size=28, color=WHITE)
        # Fix: Issue 21 - Position at E3 for better padding
        self.place_at_grid(both_label, 'E3', scale_factor=1.0)
        
        red_x = Cross(both_label, color="#FF0000", stroke_width=12)
        
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        self.play(Write(both_label))
        self.play(Create(red_x))
        self.wait(2)
