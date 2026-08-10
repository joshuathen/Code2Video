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
        self.setup_layout("Summary & Key Takeaways", [
            "CLT works for any underlying population distribution.",
            "It specifically describes the distribution of sample averages.",
            "Large samples yield highly predictable, tight bell curves."
        ])
        
        # Load assets
        pop_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg", color="#FF5733")
        sample_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sample.svg", color="#FFFFFF")
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg", color="#33FF57")
        
        pop_label = Text("Population", font_size=20, color="#FF5733")
        sample_label = Text("Sample Means", font_size=20, color="#FFFFFF")
        normal_label = Text("Normal", font_size=20, color="#33FF57")
        
        pop_group = VGroup(pop_icon, pop_label).arrange(DOWN)
        sample_group = VGroup(sample_icon, sample_label).arrange(DOWN)
        bell_group = VGroup(bell_icon, normal_label).arrange(DOWN)
        
        arrow_flow = Arrow(UP, DOWN, color="#FFFFFF")
        distribution_text = Text("Distribution of Averages", font_size=20)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(pop_group, "B3", scale_factor=0.7)
        self.play(FadeIn(pop_group))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Addressing issue 38 & 39
        self.place_at_grid(distribution_text, 'C3', scale_factor=0.75)
        self.place_at_grid(arrow_flow, 'D4', scale_factor=0.7)
        self.place_at_grid(sample_group, "E4", scale_factor=0.7)
        
        self.play(FadeIn(distribution_text), GrowArrow(arrow_flow), FadeIn(sample_group))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # Addressing issue 37
        self.place_in_area(bell_group, 'D3', 'F5', scale_factor=0.6)
        
        self.play(FadeIn(bell_group))
        self.lecture[2].set_color("#33FF57")
        
        self.wait(2)
