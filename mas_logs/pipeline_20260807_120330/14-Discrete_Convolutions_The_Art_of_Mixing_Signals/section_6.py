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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Takeaway", [
            "Convolution is the foundation of digital signal processing.",
            "It powers modern AI vision and everyday photo filters.",
            "This mathematical mixer transforms how we process information."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display 'CNN' and 'DSP' in bold cyan #00FFFF.
        self.lecture[0].set_color("#00FFFF")
        
        dsp_label = Text("DSP", weight=BOLD, color="#00FFFF")
        cnn_label = Text("CNN", weight=BOLD, color="#00FFFF")
        
        self.place_at_grid(dsp_label, "B2", scale_factor=1.0)
        self.place_at_grid(cnn_label, "B5", scale_factor=1.0)
        
        self.play(FadeIn(dsp_label), FadeIn(cnn_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Show a flow of data points entering a 'Convolution' block.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        conv_rect = Rectangle(width=2.5, height=1.2, color=BLUE_B)
        conv_text = Text("Convolution", font_size=20)
        conv_block = VGroup(conv_rect, conv_text)
        self.place_in_area(conv_block, "D3", "D4", scale_factor=0.9)
        
        # Data points
        dots = VGroup(*[Dot(radius=0.1, color=YELLOW) for _ in range(3)])
        for i, dot in enumerate(dots):
            # Start staggered
            start_pos = self.grid["D1"] + LEFT * (i * 0.8)
            dot.move_to(start_pos)
        
        self.play(FadeIn(conv_block))
        
        # Flow animation
        dot_anims = []
        for dot in dots:
            dot_anims.append(dot.animate.move_to(conv_block.get_center()).set_opacity(0))
            
        self.play(AnimationGroup(*dot_anims, lag_ratio=0.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade in labels and icons: 'Self-Driving Cars' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png], 
        # 'Photo Filters' [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg], 'AI'.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Asset Loading
        icon_cars = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        label_cars = Text("Self-Driving Cars", font_size=18)
        
        icon_filters = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg").set_color(WHITE)
        label_filters = Text("Photo Filters", font_size=18)
        
        label_ai = Text("AI", font_size=18)
        
        # Positioning according to VideoCritic (Issues 34, 35, 36)
        # Spanning two columns for width and using Row F for labels as requested
        self.place_in_area(label_cars, 'F1', 'F2', scale_factor=0.8)
        self.place_in_area(icon_cars, 'E1', 'E2', scale_factor=0.7) # Icons in Row E
        
        self.place_in_area(label_filters, 'F3', 'F4', scale_factor=0.8)
        self.place_in_area(icon_filters, 'E3', 'E4', scale_factor=0.6) # SVGs often need different scaling
        
        self.place_in_area(label_ai, 'F5', 'F6', scale_factor=0.8)
        # AI text label centered in F5-F6
        
        self.play(
            FadeIn(icon_cars), FadeIn(label_cars),
            FadeIn(icon_filters), FadeIn(label_filters),
            FadeIn(label_ai)
        )
        self.wait(3)

        # Final color reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(2)
