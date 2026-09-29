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
        self.setup_layout("Prerequisite: Cryptographic Hashing", [
            "One-way functions hide input data.",
            "Hash turns data into unique, fixed-length strings.",
            "You cannot reverse a hash to see input."
        ])
        
        # Load assets
        alice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/person.svg")
        
        # === Animation for Lecture Line 1 ===
        input_text = Text("Alice", color='#FF6F61')
        alice_label = Text("Input", font_size=20)
        hash_box = Rectangle(color='#F7786B', height=1.0, width=1.0)
        hash_label = Text("Hash", font_size=20)
        hash_func = VGroup(hash_box, hash_label)
        
        self.place_at_grid(alice_icon, 'C1', scale_factor=0.7)
        self.place_at_grid(input_text, 'D1', scale_factor=0.7)
        self.place_at_grid(alice_label, 'B1', scale_factor=0.6)
        
        self.place_at_grid(hash_func, 'C3', scale_factor=0.7)
        self.place_at_grid(hash_label, 'B3', scale_factor=0.6)
        
        arrow = Arrow(start=alice_icon.get_right(), end=hash_func.get_left(), buff=0.2)
        
        self.play(FadeIn(alice_icon), FadeIn(input_text), FadeIn(alice_label), Create(hash_func), Create(hash_label), GrowArrow(arrow))
        self.lecture[0].set_color('#FF6F61')
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        output_text = Text("a7f3...", color='#88B04B')
        hash_result_label = Text("Hash Output", font_size=20)
        self.place_at_grid(output_text, 'C5', scale_factor=0.7)
        self.place_at_grid(hash_result_label, 'B5', scale_factor=0.6)
        arrow2 = Arrow(start=hash_func.get_right(), end=output_text.get_left(), buff=0.2)
        
        self.play(GrowArrow(arrow2), FadeIn(output_text), FadeIn(hash_result_label))
        self.lecture[1].set_color('#88B04B')
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        reverse_arrow = Arrow(start=hash_func.get_left(), end=alice_icon.get_right(), color='#D64161', buff=0.2)
        cross = Cross(reverse_arrow, stroke_color='#D64161')
        
        self.play(Create(reverse_arrow), Create(cross))
        self.lecture[2].set_color('#D64161')
        self.wait(2)
