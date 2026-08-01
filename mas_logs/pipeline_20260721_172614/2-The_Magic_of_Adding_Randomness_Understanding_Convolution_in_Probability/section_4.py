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
        # Fetching content from storyboard
        title_text = "Visualizing Convolution: The 'Flip and Slide'"
        lecture_lines = [
            "For continuous variables, we use the 'Flip and Slide'.",
            "One PDF stays still while the other is flipped.",
            "The flipped PDF slides across the stationary one.",
            "The overlapping area calculates the new probability density.",
            "Watch as two squares convolve into a triangle shape."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        GREEN = "#00FF00"
        MAGENTA = "#FF00FF"
        RED = "#FF0000"
        YELLOW = "#FFFF00"
        WHITE = "#FFFFFF"
        
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/pdf.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GREEN)
        
        # Two rectangular PDF blocks using SVGMobject [Asset: pdf.svg]
        green_block = SVGMobject(asset_path).set_color(GREEN)
        magenta_block = SVGMobject(asset_path).set_color(MAGENTA)
        
        # Standardize size for consistent overlap calculation
        green_block.height = 1.0
        magenta_block.height = 1.0
        
        # Positioning: Issues 30 & 31
        self.place_at_grid(green_block, "C4", scale_factor=1.2)
        self.place_at_grid(magenta_block, "C6", scale_factor=1.2)
        
        self.play(FadeIn(green_block), FadeIn(magenta_block))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)
        
        # Mirror the magenta block horizontally and change its color to red
        self.play(
            magenta_block.animate.flip(UP).set_color(RED),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        
        # Use ValueTracker to control the sliding offset (x-distance between blocks)
        # Starting offset is 2.0 (C4 to C6)
        offset_tracker = ValueTracker(2.0)
        
        # Red block (formerly magenta) follows the tracker
        magenta_block.add_updater(lambda m: m.move_to(green_block.get_center() + RIGHT * offset_tracker.get_value()))
        
        # Slide to the left of the green block
        self.play(offset_tracker.animate.set_value(-2.0), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Overlap highlight: Rectangle whose width and position are updated
        # Block width after 1.2 scale is approx 1.2
        block_w = green_block.width
        overlap_rect = Rectangle(width=0.01, height=green_block.height, fill_opacity=0.6, color=YELLOW, fill_color=YELLOW, stroke_width=0)
        overlap_rect.move_to(green_block.get_center())
        
        def update_overlap(rect):
            off_x = offset_tracker.get_value()
            ol_w = max(0.001, block_w - abs(off_x))
            if ol_w > 0.001:
                rect.stretch_to_fit_width(ol_w, about_point=rect.get_center())
                l1, r1 = -block_w/2, block_w/2
                l2, r2 = off_x - block_w/2, off_x + block_w/2
                ol_l, ol_r = max(l1, l2), min(r1, r2)
                rect.move_to(green_block.get_center() + RIGHT * (ol_l + ol_r) / 2)
                rect.set_opacity(0.6)
            else:
                rect.set_opacity(0)
        
        overlap_rect.add_updater(update_overlap)
        self.add(overlap_rect)
        
        # Move until overlap starts to show the effect
        self.play(offset_tracker.animate.set_value(-1.0), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        # Triangle graph trackers
        # Base Y level: Row E center
        base_y = self.grid["E4"][1]
        axis = Line(self.grid["E1"], self.grid["E6"], color=WHITE, stroke_width=1)
        self.add(axis)

        # Trace point for the convolution result (area of overlap)
        trace_dot = Dot(radius=0.02, color=WHITE)
        def update_trace_dot(dot):
            off_x = offset_tracker.get_value()
            # Area of two overlapping rectangles of width block_w and height H
            # Area = max(0, block_w - |off_x|) * height
            # We scale the area visually to fit the grid
            area = max(0, block_w - abs(off_x)) * 0.8 
            dot.move_to([magenta_block.get_center()[0], base_y + area, 0])
            
        trace_dot.add_updater(update_trace_dot)
        path = TracedPath(trace_dot.get_center, stroke_color=WHITE, stroke_width=3)
        self.add(path, trace_dot)
        
        # Full convolution slide
        self.play(offset_tracker.animate.set_value(2.0), run_time=6, rate_func=linear)
        self.wait(2)
