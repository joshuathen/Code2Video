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
        lecture_lines = [
            "A hash function is a digital shredder.", 
            "Input yields a unique, fixed-length output.", 
            "Tiny input changes alter the hash completely."
        ]
        self.setup_layout("Prerequisite: The Digital Fingerprint (Hashing)", lecture_lines)
        
        # Setup objects
        input_box = Rectangle(width=1.5, height=1.0, color="#00FF00").set_fill(opacity=0.3)
        input_text = Text("Data", font_size=24, color="#00FF00").move_to(input_box.get_center())
        data_group = VGroup(input_box, input_text)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/shredder.svg]
        shredder = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shredder.svg")
        func_group = VGroup(shredder)
        
        hash_rect = Rectangle(width=2.5, height=0.5, color="#FF0000").set_fill(opacity=0.3)
        hash_val = Text("0x4a5b...", font_size=24, color="#FF0000").move_to(hash_rect.get_center())
        hash_group = VGroup(hash_rect, hash_val)
        
        # Apply layout suggestions (B22/B23)
        self.place_in_area(data_group, 'C1', 'D2', 0.75)
        self.place_in_area(func_group, 'C3', 'D4', 0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(data_group), FadeIn(func_group))
        self.lecture[0].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(hash_group, 'D6', 0.9)
        arrow = Arrow(data_group.get_right(), func_group.get_left())
        arrow2 = Arrow(func_group.get_right(), hash_group.get_left())
        self.play(GrowArrow(arrow), GrowArrow(arrow2), FadeIn(hash_group))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        new_input_text = Text("Deta", font_size=24, color="#00FF00").move_to(input_box.get_center())
        new_hash_val = Text("0x9c1d...", font_size=24, color="#FF0000").move_to(hash_rect.get_center())
        
        self.play(
            Transform(input_text, new_input_text),
            Transform(hash_val, new_hash_val)
        )
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
