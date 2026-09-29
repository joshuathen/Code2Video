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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["One head captures limited relationships.", "Multiple heads run in parallel.", "Each head learns different perspectives."]
        self.setup_layout("Multi-Head Attention: Seeing from Multiple Perspectives", lecture_lines)
        
        # Create base attention blocks
        def create_block(color):
            return Rectangle(width=1.5, height=0.8, fill_opacity=0.5, color=color, fill_color=color)

        heads = VGroup(
            create_block("#8A2BE2"),
            create_block("#FF4500"),
            create_block("#32CD32")
        ).arrange(RIGHT, buff=0.5)

        # Asset loading
        eye = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")

        # === Animation for Lecture Line 1 ===
        single_head = create_block("#8A2BE2")
        self.place_at_grid(single_head, "B2", scale_factor=0.7)
        self.play(FadeIn(single_head))
        self.play(self.lecture[0].animate.set_color("#8A2BE2"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeOut(single_head))
        self.place_in_area(heads, "B3", "D4", scale_factor=0.6)
        eye.scale(0.5).next_to(heads, UP)
        self.play(FadeIn(heads), FadeIn(eye))
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        concat_layer = Rectangle(width=5.0, height=0.5, fill_opacity=0.7, color=WHITE, fill_color=WHITE)
        self.place_at_grid(concat_layer, "E4", scale_factor=0.8)
        
        connections = VGroup()
        for head in heads:
            connections.add(Line(head.get_bottom(), concat_layer.get_top(), color=head.get_fill_color()))
            
        prism.scale(0.5).next_to(concat_layer, RIGHT)
        self.play(Create(concat_layer), Create(connections), FadeIn(prism))
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        self.wait(2)
