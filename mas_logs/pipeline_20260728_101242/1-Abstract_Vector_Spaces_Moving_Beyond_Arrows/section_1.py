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
        # Data from storyboard
        title_text = "Prerequisite: The Concrete Vector"
        lecture_lines = [
            "Traditionally, we visualize vectors as arrows in space.",
            "We can add arrows by placing them head-to-tail.",
            "We can scale arrows by stretching or shrinking them."
        ]
        
        # Colors
        color_v = "#00CCFF"
        color_w = "#00FF00"
        color_scaling = "#FFFF00"
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(color_v))
        
        # Setup Coordinate System
        axes = Axes(
            x_range=[0, 7, 1],
            y_range=[0, 5, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": True, "color": GREY_C},
            tips=False
        )
        # Position axes in the center-right area
        self.place_in_area(axes, 'B2', 'F6')
        self.add(axes)
        
        # Vector v (3, 2)
        v_arrow = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(3, 2),
            buff=0,
            color=color_v,
            stroke_width=6
        )
        # Vector label
        v_label = MathTex(r"\vec{v}", color=color_v)
        # Place label near the vector arrow
        # Fix Issue 19: Move v_label from C4 to C5 to avoid overlap with arrowhead
        self.place_at_grid(v_label, 'C5', scale_factor=0.8)
        
        self.play(Create(v_arrow), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_w)
        )
        
        # Vector w (1, 1) starting from v's tip (head-to-tail)
        w_arrow = Arrow(
            start=axes.c2p(3, 2),
            end=axes.c2p(4, 3),
            buff=0,
            color=color_w,
            stroke_width=6
        )
        w_label = MathTex(r"\vec{w}", color=color_w)
        # Position w label near its tip
        # Fix Issue 20: Move w_label from B5 to B6 to avoid overlap with arrowhead
        self.place_at_grid(w_label, 'B6', scale_factor=0.8)
        
        # Resultant vector v+w to visualize the addition result
        vw_res = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(4, 3),
            buff=0,
            color=WHITE,
            stroke_width=2
        )
        
        self.play(Create(w_arrow), Write(w_label))
        self.play(Create(vw_res))
        self.wait(2)
        
        # Cleanup addition vectors to prepare for scaling demo
        self.play(FadeOut(w_arrow), FadeOut(w_label), FadeOut(vw_res))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_scaling)
        )
        
        # Scale v by factor of 2: (6, 4)
        v_scaled_arrow = Arrow(
            start=axes.c2p(0, 0),
            end=axes.c2p(6, 4),
            buff=0,
            color=color_scaling,
            stroke_width=6
        )
        v_scaled_label = MathTex(r"2\vec{v}", color=color_scaling)
        # Position label for scaled vector
        # Fix Issue 18: Move v_scaled_label from A6 to B6 to avoid tight margins
        self.place_at_grid(v_scaled_label, 'B6', scale_factor=0.8)
        
        self.play(
            Transform(v_arrow, v_scaled_arrow),
            Transform(v_label, v_scaled_label)
        )
        self.wait(2)
