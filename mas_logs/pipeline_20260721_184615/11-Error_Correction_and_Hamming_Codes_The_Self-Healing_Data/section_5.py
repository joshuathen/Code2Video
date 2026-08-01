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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Calculation: Who Checks Whom?"
        lines = [
            "Every bit position has a unique binary address.",
            "Parity bit 1 checks addresses ending in one.",
            "Parity bit 2 checks addresses with a middle one.",
            "Each parity bit covers a specific pattern of bits.",
            "This creates a digital fingerprint for every position."
        ]
        self.setup_layout(title, lines)

        # Helper to create index rows for bits 1-7
        indices_vgroup = VGroup()
        indices_list = []
        for i in range(1, 8):
            binary_str = format(i, '03b')
            # Using simple MathTex to represent the index and its binary form
            row = MathTex(rf"{i}: ({binary_str})", color="#FFFFFF", font_size=36)
            indices_list.append(row)
            indices_vgroup.add(row)
        
        indices_vgroup.arrange(DOWN, buff=0.3)
        
        # POSITIONING: Use area A2 to F4 as suggested by Critic to bring visuals closer to lecture notes
        self.place_in_area(indices_vgroup, "A2", "F4", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # List binary indices 001 to 111 vertically in white (#FFFFFF). 
        # Change Line 1 color to #FFFF00.
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(indices_vgroup, shift=RIGHT))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Highlight rows with a '1' in the last digit in cyan (#00FFFF). 
        # Change Line 2 color to #FFFF00.
        # Last bit corresponds to P1. Positions: 1(001), 3(011), 5(101), 7(111)
        p1_indices = [0, 2, 4, 6] 
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(
            *[Indicate(indices_list[i], color="#00FFFF") for i in p1_indices],
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight rows with a '1' in the middle digit in magenta (#FF00FF). 
        # Change Line 3 color to #FFFF00.
        # Middle bit corresponds to P2. Positions: 2(010), 3(011), 6(110), 7(111)
        p2_indices = [1, 2, 5, 6] 
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(
            *[Indicate(indices_list[i], color="#FF00FF") for i in p2_indices],
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Highlight rows with a '1' in the first digit in yellow (#FFFF00). 
        # Change Line 4 color to #FFFF00.
        # First bit corresponds to P3. Positions: 4(100), 5(101), 6(110), 7(111)
        p3_indices = [3, 4, 5, 6] 
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        self.play(
            *[Indicate(indices_list[i], color="#FFFF00") for i in p3_indices],
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Highlight index 3 (011) to show it's within both cyan and magenta groups. 
        # Change Line 5 color to #FFFF00.
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        
        idx_3 = indices_list[2] # 0-indexed position for bit 3
        # Use a surrounding rectangle and sequential pulses to show the intersection
        highlight_box = SurroundingRectangle(idx_3, color="#FFFFFF", buff=0.1)
        self.play(Create(highlight_box))
        # Belonging to P1 (Cyan)
        self.play(Indicate(idx_3, color="#00FFFF", scale_factor=1.2), run_time=1)
        # Belonging to P2 (Magenta)
        self.play(Indicate(idx_3, color="#FF00FF", scale_factor=1.2), run_time=1)
        self.wait(2)
