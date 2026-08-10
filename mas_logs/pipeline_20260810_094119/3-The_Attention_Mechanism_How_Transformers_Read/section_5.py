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
        lecture_lines = [
            "One attention head is rarely enough.",
            "Parallel heads capture different types of relationships.",
            "One head tracks grammar, another tracks tone.",
            "Combining heads provides a complete understanding.",
            "Multi-head attention mimics human cognition."
        ]
        self.setup_layout("Multi-Head Attention: Seeing the Whole Picture", lecture_lines)
        
        # Load Assets
        brain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/brain.svg")
        
        # Create heads
        head_colors = ["#FF5733", "#33FF57", "#3357FF", "#FFFF33"]
        heads = VGroup(*[Square(side_length=1.5, color=c) for c in head_colors])
        head_labels = VGroup(*[Text(f"Head {i+1}", font_size=16) for i in range(4)])
        attention_group = VGroup(*[VGroup(heads[i], head_labels[i]) for i in range(4)]).arrange(RIGHT)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(brain_icon, "A2", "A5", scale_factor=0.3)
        self.place_in_area(attention_group, "B2", "B5", scale_factor=0.7)
        self.play(FadeIn(brain_icon), Create(heads), Write(head_labels))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(*[Indicate(heads[i]) for i in range(4)])
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        self.play(
            heads[0].animate.set_color("#FF5733"),
            heads[1].animate.set_color("#33FF57")
        )
        self.lecture[2].set_color("#FFFFFF")

        # === Animation for Lecture Line 4 ===
        concat_bar = Rectangle(height=0.5, width=4, color=WHITE)
        self.place_at_grid(concat_bar, "C3", scale_factor=0.5)
        self.play(FadeIn(concat_bar))
        self.lecture[3].set_color("#FFFFFF")

        # === Animation for Lecture Line 5 ===
        output_vec = Arrow(start=ORIGIN, end=RIGHT*2, color="#CCCCCC")
        self.place_at_grid(output_vec, "E3", scale_factor=0.6)
        
        # Flash brain icon
        self.play(
            GrowArrow(output_vec),
            brain_icon.animate.set_color("#FFFFFF")
        )
        self.lecture[4].set_color("#FFFFFF")
        self.wait(2)
