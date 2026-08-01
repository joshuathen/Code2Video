from manim import *
import numpy as np

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
        # Fetching data from shared state
        title = "Self-Similarity and the Infinite Zoom"
        lines = [
            "These boundaries reveal intricate, repeating fractal structures.",
            "Zooming in reveals tiny, perfect copies of the whole.",
            "Local shapes are preserved while global views are distorted.",
            "This infinite self-similarity defines the beauty of holomorphic dynamics.",
            "Simple rules generate infinite, self-repeating complexity."
        ]
        self.setup_layout(title, lines)

        def get_mandelbrot_set(color=BLUE_B):
            # Simplified geometric representation of the Mandelbrot set
            # Main cardioid: x = 1/2*cos(t) - 1/4*cos(2t), y = 1/2*sin(t) - 1/4*sin(2t)
            cardioid = ParametricFunction(
                lambda t: np.array([
                    0.5 * np.cos(t) - 0.25 * np.cos(2 * t),
                    0.5 * np.sin(t) - 0.25 * np.sin(2 * t),
                    0
                ]),
                t_range=[0, TAU],
                color=color
            ).scale(1.5).shift(LEFT * 0.2)
            
            # Main bulb at the left (period-2)
            bulb_left = Circle(radius=0.35, color=color).next_to(cardioid, LEFT, buff=-0.1)
            
            # Top and bottom bulbs
            bulb_top = Circle(radius=0.15, color=color).shift(UP * 0.8 + LEFT * 0.4)
            bulb_bottom = Circle(radius=0.15, color=color).shift(DOWN * 0.8 + LEFT * 0.4)
            
            # Main antenna
            antenna = Line(start=bulb_left.get_left(), end=bulb_left.get_left() + LEFT*0.5, color=color)
            
            mset = VGroup(cardioid, bulb_left, bulb_top, bulb_bottom, antenna)
            return mset

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_B)
        main_mset = get_mandelbrot_set(BLUE_B)
        # Fix Issue 28: Scale factor reduced to 1.0 to avoid overflow
        self.place_in_area(main_mset, "A1", "F6", scale_factor=1.0)
        
        focus_arrow = Arrow(start=RIGHT, end=LEFT, color=YELLOW, buff=0.1).scale(0.6)
        # Position arrow near the left bulb
        focus_arrow.move_to(main_mset[1].get_left() + RIGHT*0.2 + UP*0.5)
        
        self.play(Create(main_mset), run_time=2)
        self.play(GrowArrow(focus_arrow))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        # Zoom into the left bulb
        zoom_point = main_mset[1].get_center()
        
        self.play(
            FadeOut(focus_arrow),
            main_mset.animate.scale(8, about_point=zoom_point),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN_A)
        
        # Reveal a Mini-Mandelbrot identical to the original
        mini_mandel = get_mandelbrot_set("#00FF00")
        mini_mandel.scale(0.12).move_to(zoom_point)
        
        self.play(FadeIn(mini_mandel, scale=0.5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF00FF")
        
        # Create spiral structures around the mini-mandel
        spirals = VGroup()
        for i in range(5):
            angle = i * TAU / 5
            spiral = ParametricFunction(
                lambda t: np.array([
                    0.03 * t * np.cos(t),
                    0.03 * t * np.sin(t),
                    0
                ]),
                t_range=[0, TAU*2.5],
                color="#FF00FF"
            ).scale(0.3).rotate(angle).move_to(zoom_point + 0.3 * np.array([np.cos(angle), np.sin(angle), 0]))
            spirals.add(spiral)
            
        self.play(Create(spirals), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        inf_text = Text("Infinite Complexity", font_size=26, color=WHITE)
        # Fix Issue 30: Repositioned to avoid overlap with spirals and adjusted scale
        self.place_in_area(inf_text, 'E5', 'F6', scale_factor=0.75)
        
        self.play(Write(inf_text))
        
        # Slow pulse of the mini-mandel and spirals to emphasize complexity
        self.play(
            mini_mandel.animate.scale(1.1),
            spirals.animate.scale(1.1).rotate(0.2),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
