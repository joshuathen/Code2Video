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
        lecture_lines = ["Binary uses only zero and one.", "Each digit represents powers of two.", "Example: five is one-zero-one."]
        self.setup_layout("Prerequisite: The Binary Language", lecture_lines)
        
        # Note: Assets are none.svg placeholders per prompt, will use shape objects.
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] -> Square
        bit0 = Square(side_length=0.7, color="#FF5733", fill_opacity=0.5)
        text0 = Text("0", font_size=36).move_to(bit0.get_center())
        bit_group = VGroup(bit0, text0)
        self.place_at_grid(bit_group, 'C2', scale_factor=0.8)
        self.play(Create(bit_group), FadeToColor(self.lecture[0], "#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        bit1 = Square(side_length=0.7, color="#33FF57", fill_opacity=0.5)
        text1 = Text("1", font_size=36).move_to(bit1.get_center())
        bit_group_1 = VGroup(bit1, text1)
        self.place_at_grid(bit_group_1, 'C3', scale_factor=0.8)
        self.play(FadeIn(bit_group_1), FadeToColor(self.lecture[1], "#33FF57"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg] -> Rounded Rectangle
        binary_seq = Text("101", font_size=48, color="#3357FF")
        self.place_at_grid(binary_seq, 'C4', scale_factor=0.8)
        self.play(Write(binary_seq), FadeToColor(self.lecture[2], "#3357FF"))
        self.wait(1)
