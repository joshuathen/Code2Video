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
        self.setup_layout(
            "Prerequisite: One-Way Hash Functions", 
            [
                "Hash functions act like a digital meat grinder.",
                "They turn unique inputs into fixed-length strings.",
                "Going from input to hash is computationally simple.",
                "Reversing a hash to find the input is impossible."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        # Show 'Hash Function' box #FFD700 in center with 'Input' arrow.
        hash_box = Rectangle(width=2.5, height=1.5, color="#FFD700", stroke_width=4)
        hash_label = Text("Hash Function", font_size=20, color="#FFD700")
        hash_vg = VGroup(hash_box, hash_label)
        self.place_in_area(hash_vg, "C3", "D4")
        
        # Arrow from Input side to Box
        in_arrow = Arrow(self.grid["C1"], self.grid["C3"], buff=0.1, color=WHITE)
        in_tag = Text("Input", font_size=18, color=WHITE).next_to(in_arrow, UP, buff=0.1)
        
        self.play(
            Create(hash_vg),
            GrowArrow(in_arrow),
            Write(in_tag),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        # Move 'Secret Key' text #FFFFFF into box; '8f2b...' text #00FFFF exits.
        # ISSUE 25 FIX: Move input_text to C2
        input_text = Text("Secret Key", font_size=18, color=WHITE)
        self.place_at_grid(input_text, "C2")
        
        # ISSUE 26 FIX: Move initial position of output_text to D5
        output_text = Text("8f2b7a9...", font_size=18, color="#00FFFF")
        self.place_at_grid(output_text, "D5") 
        output_text.set_opacity(0)
        
        out_arrow = Arrow(self.grid["D4"], self.grid["D6"], buff=0.1, color="#00FFFF")
        out_tag = Text("Hash", font_size=18, color="#00FFFF").next_to(out_arrow, UP, buff=0.1)

        self.play(
            input_text.animate.move_to(hash_vg.get_center()).set_opacity(0),
            run_time=1.5
        )
        self.play(
            GrowArrow(out_arrow),
            Write(out_tag),
            output_text.animate.move_to(self.grid["D6"]).set_opacity(1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        # Show an 'Easy' checkmark #00FF00 next to the arrow pointing from Input to Hash.
        # ISSUE 27 FIX: Move checkmark to B3
        checkmark = Text("✔ Easy", font_size=22, color="#00FF00")
        self.place_at_grid(checkmark, "B3")
        
        self.play(FadeIn(checkmark, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF0000")
        # Show red #FF0000 'X' over a reverse arrow from Hash back to Input.
        rev_arrow = Arrow(self.grid["E6"], self.grid["E1"], color=RED, buff=0.1)
        x_mark = Text("✘ Impossible", font_size=22, color="#FF0000")
        self.place_in_area(x_mark, "E3", "E4") # Centered under the process
        
        self.play(
            GrowArrow(rev_arrow),
            Write(x_mark),
            run_time=1.5
        )
        self.wait(2)
