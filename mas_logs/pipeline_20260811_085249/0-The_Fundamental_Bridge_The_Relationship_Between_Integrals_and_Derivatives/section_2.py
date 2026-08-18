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
        self.setup_layout("The Conceptual Inverse: Undo-ing the Action", [
            "Differentiation and integration are inverse processes.",
            "Think of them like addition and subtraction.",
            "They undo each other completely."
        ])
        
        # Setup visualization elements using SVGs
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg]
        def create_box_with_label(label_text, color=WHITE):
            box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg")
            box.set_color(color)
            label = MathTex(label_text, font_size=40, color=color)
            return VGroup(box, label)

        f_group = create_box_with_label("f(x)", WHITE)
        self.place_at_grid(f_group, "B2", scale_factor=0.7)
        
        d_group = create_box_with_label("D", "#FF00FF")
        self.place_in_area(d_group, "B3", "B4", scale_factor=0.6)
        
        fp_label = MathTex("f'(x)", font_size=36)
        self.place_at_grid(fp_label, "B6", scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(f_group))
        self.play(Create(d_group), Write(fp_label))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Adding integration box
        i_group = create_box_with_label("I", "#FF00FF")
        self.place_in_area(i_group, "D3", "D4", scale_factor=0.6)
        self.play(Create(i_group))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        # Reversing operation
        arrow1 = Arrow(start=f_group.get_right(), end=d_group.get_left(), color=WHITE)
        arrow2 = Arrow(start=d_group.get_right(), end=fp_label.get_left(), color=WHITE)
        arrow3 = Arrow(start=fp_label.get_bottom(), end=i_group.get_right(), color=WHITE)
        arrow4 = Arrow(start=i_group.get_left(), end=f_group.get_bottom(), color=WHITE)
        self.play(Create(arrow1), Create(arrow2), Create(arrow3), Create(arrow4))
        self.wait(2)
