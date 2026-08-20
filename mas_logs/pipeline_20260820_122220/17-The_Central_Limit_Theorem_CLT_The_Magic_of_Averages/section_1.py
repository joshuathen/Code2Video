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
            "A sampling distribution is a collection of sample means.",
            "We shift focus from individual data to averages.",
            "Each sample provides a single average value."
        ]
        self.setup_layout("Prerequisite: The Sampling Distribution", lecture_lines)
        
        # Assets
        pop_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg"
        
        # === Animation for Lecture Line 1 ===
        # Create a large population of random values
        pop_icon = SVGMobject(pop_svg, color=WHITE)
        self.place_in_area(pop_icon, 'A4', 'C6', scale_factor=0.5)
        self.play(FadeIn(pop_icon))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Show mean of population as a fixed line
        mean_line = Line(start=self.grid['D4'], end=self.grid['D6'], color="#FFFF00", stroke_width=4)
        self.play(Create(mean_line))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Take multiple samples and calculate means
        sample_dots = VGroup(*[Dot(color="#00FFFF") for _ in range(5)])
        self.place_in_area(sample_dots, 'D4', 'F6', scale_factor=0.5)
        mean_val = Dot(color="#FF00FF")
        self.place_at_grid(mean_val, 'E5', scale_factor=0.8)
        
        self.play(FadeIn(sample_dots))
        self.play(FadeIn(mean_val))
        self.lecture[2].set_color("#00FFFF")
        
        # Plot means to form the sampling distribution (referenced in storyboard)
        dist_icon = SVGMobject(pop_svg, color="#00FF00")
        self.place_at_grid(dist_icon, 'F6', scale_factor=0.3)
        self.play(FadeIn(dist_icon))
