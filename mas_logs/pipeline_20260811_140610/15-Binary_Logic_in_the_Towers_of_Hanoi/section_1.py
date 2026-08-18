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
        self.setup_layout("Prerequisite: The Nature of Binary", 
                          ["Binary uses 0 and 1.", 
                           "Each position represents powers of two.", 
                           "Counting follows a specific sequence."])
        self.lecture.set_opacity(0)
        
        # === Animation for Lecture Line 1 ===
        # Binary uses 0 and 1.
        self.play(FadeIn(self.lecture[0]))
        zero = Text("0", color="#007FFF", font_size=48)
        one = Text("1", color="#007FFF", font_size=48)
        # Fix 22: Binary digits positioning
        self.place_at_grid(zero, 'B3', scale_factor=0.7)
        self.place_at_grid(one, 'B4', scale_factor=0.7)
        # Asset 1: switch.svg
        switch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg")
        self.place_at_grid(switch, 'B5', scale_factor=0.7)
        self.play(Write(zero), Write(one), FadeIn(switch))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Each position represents powers of two.
        self.play(FadeIn(self.lecture[1]))
        pos_labels = VGroup(*[Text(f"2^{i}", color=WHITE, font_size=24) for i in range(3)])
        # Fix 21: Pos labels positioning
        self.place_in_area(pos_labels, 'D3', 'D5', scale_factor=0.6)
        # Asset 2: counter.svg
        counter = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/counter.svg")
        self.place_at_grid(counter, 'D6', scale_factor=0.6)
        self.play(FadeIn(pos_labels), FadeIn(counter))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Counting follows a specific sequence.
        self.play(FadeIn(self.lecture[2]))
        # Sequence: 000, 001, 010, 011, 100
        sequence = ["000", "001", "010", "011", "100"]
        display = Text(sequence[0], color=YELLOW, font_size=48)
        # Fix 23: Display positioning
        self.place_at_grid(display, 'E4', scale_factor=0.8)
        self.add(display)
        
        for i in range(1, len(sequence)):
            new_display = Text(sequence[i], color=YELLOW, font_size=48)
            self.place_at_grid(new_display, 'E4', scale_factor=0.8)
            self.play(Transform(display, new_display))
            self.wait(0.5)
        
        self.wait(2)
